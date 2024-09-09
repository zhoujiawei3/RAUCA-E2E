
import argparse
import logging
import os
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch.utils.data
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw
from models.yolo import Model
from utils.datasets_RAUCA import create_dataloader
from utils.general_RAUCA import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, colorstr
from utils.google_utils import attempt_download
from utils.loss_RAUCA import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import neural_renderer
from PIL import Image,ImageOps
from Image_Segmentation.network import U_Net
import torch.nn.functional as F
import kornia.augmentation as K
import torchvision.transforms as Tr
import kornia.geometry as KG
import torchvision.models as models
from neural_renderer.load_obj_differentiable_new import LoadTextures
logger = logging.getLogger(__name__)

# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))
# WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# GIT_INFO = check_git_info()
def mix_image(image_optim, mask,origin_image):
    
    return (1 - mask) * origin_image + mask *torch.clamp(image_optim,0,1)
    return origin_image
def flip_image(image_path, saved_location):
    """
    Flip or mirror the image

    @param image_path: The path to the image to edit
    @param saved_location: Path to save the flipped image
    """
    image_obj = Image.open(image_path)
    flipped_image = ImageOps.flip(image_obj)
    flipped_image.save(saved_location)

class ROA(torch.nn.Module):
    def __init__(self):
        super(ROA, self).__init__()
        self.randomAffine = K.RandomAffine(degrees=(-5.0, 5.0), translate=(0.1, 0.1),scale=(0.7,1.0),keepdim=True)
        # K.RandomResizedCrop((640, 640)),
        self.colorJiggle = K.ColorJiggle(0.25, (0.75,1.5),p=0.5,keepdim=True)
            #K.RandomAffine(degrees=0,scale=(0.1, 0.2)),

            #K.RandomHorizontalFlip(),
        
            #KG.transform.rotate(torch.tensor([0, 90, 180, 270])),
        

    def forward(self, x):
        
        x = self.randomAffine(x)
        
    #判断x是否是NaN充满
        if not torch.isnan(x).any():
            x = self.colorJiggle(x)

        return x
def draw_red_origin(file_path):
    # 打开图像文件
    image = Image.open(file_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 创建一个新的图像对象，用于绘制点
    new_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(new_image)

    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 绘制红色的原点（半径为3个像素）
    radius = 3
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=(255, 0, 0))

    # 合并原始图像和绘制的点
    print(new_image.size,image.convert('RGBA').size)
    result_image = Image.alpha_composite(image.convert('RGBA'), new_image)

    # 保存结果图像
    result_file_path = file_path
    result_image.save(result_file_path)

    return result_file_path

def loss_smooth(img, mask):
    # 平滑损失 ==> 使得在边界的对抗性扰动不那么突兀，更加平滑
    # [1,3,223,23]
    print(f"img.shape:{img.shape}")
    print(f"mask.shape:{mask.shape}")
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2) #xi,j − xi+1,j
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2) #xi,j − xi,j+1
    # [3,223,223]
    mask = mask[:, :,:-1, :-1]

    # mask = mask.unsqueeze(1)
    return T * torch.sum(mask * (s1 + s2)) #论文中的μ
def loss_smooth_UVmap(img, mask):
    # print(img.shape)
    # print(mask.shape)
    s1 = torch.pow(img[ 1:, :-1, :] - img[:-1, :-1, :], 2) #xi,j − xi+1,j
    s2 = torch.pow(img[ :-1, 1:, :] - img[ :-1, :-1, :], 2) #xi,j − xi,j+1
    # [3,223,223]
    mask = mask[ :-1,:-1, :]
    
    # mask = mask.unsqueeze(1)
    return T * torch.sum(mask * (s1 + s2))/2048.0/2048.0 #论文中的μ

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):
    # 计算纹理
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)# torch.nn.Tanh()()双曲正切函数，结果让纹理的参数在-1到1，加一乘0.5是让他在0-1之间
    return texture_origin * (1 - texture_mask) + texture_mask * textures  #这里让纹理参数作用到掩码为1的位置上，加上前面哪项，是让剩余不想被对抗性纹理影响的点是原来图像的面片


def train(hyp, opt, device):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank


    
    # ---------------------------------#
    # -------Load 3D model-------------#
    # ---------------------------------#
    texture_size = opt.texturesize #这里的纹理就是T，那个2D的奇形怪状的东西
    vertices, faces, texture_origin,image,faces_3_2,is_update,wrap_way,bilinear_way = neural_renderer.load_obj_totalUV(filename_obj=opt.obj_file, texture_size=texture_size,load_texture=True)  
    print(f"texuture_origin_shape:{texture_origin.shape}")
    print(f"image_shape:{image.shape}")
    # texuture_origin_shape:torch.Size([23145, 6, 6, 6, 3])
    # image_shape:torch.Size([1280, 1280, 3])
    # textures=LoadTextures.apply(image, faces_3_2, texture_origin, is_update,wrap_way,bilinear_way)
    # if torch.equal(textures.cuda(),texture_origin.cuda()):
    #     print("textures is equal")
    # vertices_compare, faces_compare, texture_origin_compare = neural_renderer.load_obj(filename_obj=opt.obj_file, texture_size=texture_size,load_texture=True)  
    # #看是否相同
    
    # if torch.equal(vertices, vertices_compare):
    #     print("vertices is equal")
    # if torch.equal(faces, faces_compare):
    #     print("faces is equal")
    # if torch.equal(texture_origin, texture_origin_compare):
    #     print("texture_origin is equal")
    
    #这里的obj_file应该就是3D车模，vertices顶点即物体所有多面体的顶点。维度是 [num_vertices, xyz] 每个顶点的xyz三维坐标 
    #faces面片是顶点之间的组合关系。[num_faces, v123] 面片数量和每个面片的三个顶点。v1，v2,v3这个是一个索引，即该面片在整个顶点变量的之中的第几个。每三个顶点组合在一起，形成了一个面片。
    #texture贴图：xyz贴图与面片一一对应，即根据贴图图像生成的针对每个面片的RGB值。
    #忙猜这里让texture_size是未了将扁片分成纹理片大小
    # texture_255=np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32') 
    # texture_255 = torch.autograd.Variable(torch.from_numpy(texture_255).to(device), requires_grad=False) #把这个变量自动优化

    #faces.shape:torch.Size([23145, 3])
    #texture_origin:torch.Size([23145, 6, 6, 6, 3])
    print(f"vertices.shape:{vertices.shape}")                                                           
    print(f"faces.shape:{faces.shape}")
    print(f"texture_origin.device:{texture_origin.device}")
    print(device)
    # image=image.unsqueeze(0)
    # load face points
    if opt.continueFrom!=0:
        # adv_path=f"logs_7_2_totalUV/{texture_dir_name}/texture_{opt.continueFrom}.npy"
        adv_path=f"logs/{texture_dir_name}/texture_{opt.continueFrom}.npy"
        texture_param = np.load(adv_path)
        image_optim = torch.from_numpy(texture_param).to(device).requires_grad_(True)
        epochs+=opt.continueFrom
    else:
        if opt.patchInitial=="half":
            # texture_param = np.zeros((1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32') #face.shape[0]是面的数量
            # texture_param = (texture_param * 2) - 1
            # # 将 [-1, 1] 的范围扩展到 (-inf, inf)
            # texture_param = texture_param * 3
            image_optim = (torch.ones_like(image)/2.0).to(device).requires_grad_(True)
        elif opt.patchInitial=="random":
            # texture_param = np.random.random((1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32') #face.shape[0]是面的数量
            # texture_param = (texture_param * 2) - 1
            image_optim = torch.rand_like(image).to(device).requires_grad_(True)
            # 将 [-1, 1] 的范围扩展到 (-inf, inf)
            # texture_param = texture_param * 3
        elif opt.patchInitial=="origin":
            # texture_param=texture_origin.clone().cpu().numpy()
            image_optim=image.clone().to(device).requires_grad_(True)
    image_origin=image.clone().to(device).requires_grad_(False).detach()
    # print(texture_param)
    optim = torch.optim.Adam([image_optim], lr=opt.lr)
    # texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    # with open(opt.faces, 'r') as f:
    #     face_ids = f.readlines()
    #     for face_id in face_ids:
    #         if face_id != '\n':
    #             texture_mask[int(face_id) - 1, :, :, :,
    #             :] = 1  # adversarial perturbation only allow painted on specific areas，那个face文件记载了哪些面片 这个掩码就是控制只能在某些面片上加对抗性纹理，因此有些要位置要设为0，保证其上没有梯度
    # texture_mask = torch.from_numpy(texture_mask).to(device).unsqueeze(0) #unsqueeze（0）：假设原始张量的形状为 (3, 4, 5)，则使用 unsqueeze(0) 操作后，形状变为 (1, 3, 4, 5)。其中，新增的维度作为扩展的第一个维度，大小为 1。
    mask_image_dir="car_asset_E2E/mask.png"
    mask_image = Image.open(mask_image_dir)#.convert("L")
    
    mask_image = (np.transpose((np.array(mask_image)[:,:,:3])[::-1, :, :],(0,1,2))/255).astype('uint8')
    mask_image = torch.from_numpy(mask_image).to(device)
    mask_image=  mask_image
    mask_dir = os.path.join(opt.datapath, 'masks/')

    # ---------------------------------#
    # -------Yolo-v3 setting-----------#
    # ---------------------------------#
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)  #sort_keys是是否按字母排序，如果false就不会，把hyp写到hyp.yaml里面
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict #data_dict载入的是carla。yaml
    #这部分的目的就是载入这个wandblogge
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
    #导入数据集的一些参数
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    with torch_distributed_zero_first(rank):#这一行是只让主进程执行的意思
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    #print(f"ckpt['model']:{ckpt['model']}")
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys 目前是没有anchors的被注释掉了
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect，这个是按照key取交集和exclude，最终取值是按照第一个参数的value赋值
    model.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report 这里说明了会排除掉anchors的超参数
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # EMqa
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    if pretrained:
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt


    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)，         
    #det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module 可以说明这个模型不是一个并行模型

    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])检测模型通常由多个层组成，每一层都负责不同的功能，例如提取特征、生成候选框、计算目标类别和边界框等
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    print(f"rank:{rank}")
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) #这是默认让所有cuda设备运行DP模式

    # ---------------------------------#
    # -------Load dataset-------------#
    # ---------------------------------#
    print(f"train_path:{train_path}")
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, faces, texture_size, vertices, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            prefix=colorstr('train: '), mask_dir=mask_dir, ret_mask=True)##这一步让数据集中既有图像又有mask

    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
    # ---------------------------------#
    # -------Yolo-v3 setting-----------#
    # ---------------------------------#
    # textures_255_in = cal_texture(texture_255, texture_origin, texture_mask)
    # dataset.set_textures_255(textures_255_in)
    nb = len(dataloader)  # number of batches
    print(f"nb:{nb}")
    # Model parameters  这里做的是将原先的超参数的box。cls。obj根据检测层个数做出一些调整
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)  Intersection over Union权重比率，如果obj_loss不是1，就由iou操控比率
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights 80个权重类别不同的权重，用所有labels的频率倒数表示的
    model.names = names

    # Start training
    t0 = time.time()
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    compute_loss = ComputeLoss(model)  # init loss class
    # ---------------------------------#
    # ------------Training-------------#
    # ---------------------------------#
    model_nsr=U_Net()
    
    saved_state_dict = torch.load('NRP_weights_no_car_paint/car/model_nsr_l39.pth')  # 原始的参数字典

# 假设参数是使用DistributedModel保存的
# 如果原始模型是使用DataParallel进行分布式训练，可以使用以下代码来修复参数字典的键名
    
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:]  # 去掉 'module.' 前缀
        new_state_dict[name] = v
    saved_state_dict = new_state_dict
    model_nsr.load_state_dict(saved_state_dict)
    model_nsr.to(device)
    model_ROA=ROA()
    model_ROA.to(device)
    model_ROA.eval()
    epoch_start=1+opt.continueFrom
    net = torch.hub.load('yolov3',  'custom','yolov3_9_5.pt',source='local')  
    net.eval()
    net = net.to(device)
    # model=None
    for epoch in range(epoch_start, epochs+1):  # epoch ------------------------------------------------------------------
        
        model_nsr.eval()
    # 获取第一个批次的数据
    #     batch = next(iter(dataloader))

    # #    查看批次数据的维度
    #     print(f"batch.shape:{batch.shape}")  # 使用 torch.Tensor 的 shape 方法
        pbar = enumerate(dataloader)
        # print(f"dataloader.dtype:{dataloader.dtype}")
        # print(f"texture_origin.device:{texture_origin.device}")
        # print(f"texture_param.device:{texture_param.device}")
        # print(f"texture_mask.device:{texture_mask.device}")
        image_optim_in = mix_image(image_optim, mask_image, image_origin)
        textures=LoadTextures.apply(image_optim_in, faces_3_2, texture_origin, is_update,wrap_way,bilinear_way)
        textures=textures.unsqueeze(0)
        # print(f"textures.shape_render_before:{textures.shape}")
        dataset.set_textures(textures) #这一步可以说非常诡异，把这个纹理片图输入进去，就让数据集中多了texture_img 。masks是在构建数据集的时候就塞进去了，texture_img是贴上了纹理的汽车模型的一定视角的图像
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'a_ls', 's_ls','t_loss','labels','tex_mean','grad_mean'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        # model.eval() #这一步其实就是说明了不需要用这个改变模型参数，而是改变纹理生成的参数也就是 texture_param
        #print(dataloader)
        
        mloss = torch.zeros(1, device=device)
        s_mloss=torch.zeros(1)
        a_mloss=torch.zeros(1)
        for i, (imgs, texture_img, masks,imgs_cut, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            
            # print(imgs.shape)
            # print(texture_img.shape)
            # print(masks.shape)
            # print(imgs_cut.shape)
            # print(targets.shape)
            # uint8 to float32, 0-255 to 0.0-1.0

            #TEST
            imgs_cut = imgs_cut.to(device, non_blocking=True).float() / 255.0
            imgs_in= imgs_cut[0]*masks[0]+imgs[0]*(1-masks[0])/ 255.0 
            out_tensor = model_nsr(imgs_cut)
            sig = nn.Sigmoid()
             # forward
            out_tensor=sig(out_tensor)
            tensor1 = out_tensor[:,0:3, :, :]
            tensor2 = out_tensor[:,3:6, :, :]
            
            # print(tensor1.shape)
            # print(tensor2.shape)
            
            tensor3=torch.clamp(texture_img*tensor1+tensor2,0,1)
        
            masks=masks.unsqueeze(1).repeat(1, 3, 1, 1)
            
            # imgs=(1 - masks) * imgs +(255 * tensor3) * masks
            # imgs = imgs.to(device, non_blocking=True).float() / 255.0 
            # out, train_out = model(imgs)  # forward
            # texture_img_np = 255*(imgs.detach()).data.cpu().numpy()[0]
            # texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            # print(f"texture_img_np:{texture_img_np}")
            # imgs_show=net(texture_img_np)
            # imgs_show.save(log_dir)
            # # compute loss
            # print("out shape")
            # print(out.shape)
            # print(targets.shape)
            imgs=(1 - masks) * imgs +(255 * tensor3) * masks
            background=((1 - masks) * imgs +0 * masks)/255.0
        #imgs=(1 - masks) * imgs +(255 * tensor3) * masks
            imgs = imgs.to(device, non_blocking=True).float() / 255.0 
            
            imgs_ROA=model_ROA(imgs)

            #保存imgs
            # +imgs_ROA_gaussian=add_gaussian_noise(imgs_ROA,mean=0.5,std=0.1,probablity=0.5)
            out, train_out = net.model(imgs_ROA)  # forward
        #print(f"imgs:{imgs.shape}")
            if rank in [-1, 0]:
                if i%10==0: 
                    texture_img_np = 255*(imgs_ROA.detach()).data.cpu().numpy()[0]
                    texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
                    # print(f"texture_img_np:{texture_img_np}")#texture_img_np=np.ascontiguousarray(texture_img_np)
                    # # if rank in [-1, 0]: 
                    # imgs_show=net(texture_img_np)
                    # imgs_show.save(log_dir)
                    imgs_show=net(texture_img_np)
                    #print(imgs_show)
                    imgs_show.save_ZJW(log_dir,"adversarial_with_DATAtrans_detect.png")
                    imgs_origin=(1 - masks) * imgs +( tensor3) * masks
                    texture_img_np_no_trans=255*(imgs_origin.detach()).data.cpu().numpy()[0]
                    texture_img_np_no_trans = Image.fromarray(np.transpose(texture_img_np_no_trans, (1, 2, 0)).astype('uint8'))
                    imgs_show_no_trans=net(texture_img_np_no_trans)
                    #print(imgs_show)
                    imgs_show_no_trans.save_ZJW(log_dir,"adversarial_without_DATAtrans_detect.png")

                    texture_img_np_origin_photo=255*(imgs_in.detach()).data.cpu().numpy()
                    texture_img_np_origin_photo = Image.fromarray(np.transpose(texture_img_np_origin_photo, (1, 2, 0)).astype('uint8'))
                    imgs_show_origin_photo=net(texture_img_np_origin_photo)
                    #print(imgs_show)
                    imgs_show_origin_photo.save_ZJW(log_dir,"origin_detect.png")
            
            loss1 = compute_loss(out, targets.to(device)) #这里是论文中三个对抗损失的计算区域


            
            # print(f"loss_items:{loss_items}")
            # print(f"train_out:{train_out}")
            # print(f"loss_items:{loss_items}")
            image_optim_in = mix_image(image_optim, mask_image, image_origin)
            
            loss2 = loss_smooth_UVmap(image_optim, mask_image) #这里是计算平滑损失，masks目的只计算掩码范围内的平滑损失
            loss = loss1 +loss2 #论文中的μ放在loss_smooth里
            # Backward
            
            optim.zero_grad()
            loss.backward(retain_graph=False) #retain_graph=True 参数的作用是保留计算图，以便后续可能需要进行额外的反向传播操作，这一步只是为了后续能够访问texture_param.grad
            optim.step()
            # pbar.set_description('Loss %.8f' % (loss.data.cpu().numpy()))# loss.data.cpu().numpy() 将损失张量转换为 NumPy 数组，并将其值提取出来，提取的是loss
            # print("tex mean: {:5f}, grad mean: {:5f},".format(torch.mean(texture_param).item(),
            #                                                   torch.mean(texture_param.grad).item()))
            try:
                if i%10==0:
                #Image.fromarray可以将array变成图像
                #imgs.data.cpu().numpy()[0]这是imgs里面第一张图像，乘255是缩放，np.transpose(..., (1, 2, 0))：将数组的维度从 (C, H, W) 转置为 (H, W, C)，使得通道数（例如 RGB）在数组的最后一个维度。
                    Image.fromarray(np.transpose(255 * imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'test_total.png')) 
                    
                    Image.fromarray(np.transpose(255 * imgs_ROA.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'test_total_ROA.png')) 
                    Image.fromarray(np.transpose(255 * background.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'background.png')) 
                    Image.fromarray(
                        (255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'texture2.png')) #和上面一个意思，这人非要搞两种
                    #masks0-1,1-0
                    masks_0_1 = torch.abs(1 - masks)
                    Image.fromarray(np.transpose(255 * masks_0_1.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'mask_car.png'))
                    #Image.fromarray(
                    #     (255 * imgs_ref).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                    #     os.path.join(log_dir, 'texture_ref.png'))
                    Image.fromarray(
                    (255 * imgs_cut).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                    os.path.join(log_dir, 'img_cut.png'))
                    
                    Image.fromarray(
                        (255 * tensor3).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'img_tensor3.png'))
                    Image.fromarray(
                        (255 * tensor1).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'img_tensor1.png'))
                    Image.fromarray(
                        (255 * tensor2).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, 'img_tensor2.png'))
                    Image.fromarray(np.transpose(255*imgs_in.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, '输入原图.png')) 
                    Image.fromarray(
                        (255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        os.path.join(log_dir, '渲染图.png'))
                    
                    
            except:
                pass
            
            # draw_red_origin(os.path.join(log_dir, 'test_total.png'))
            # draw_red_origin(os.path.join(log_dir, 'texture2.png'))
            # draw_red_origin(os.path.join(log_dir, 'mask.png'))
            if rank in [-1, 0]: 
                a_mloss=(a_mloss*i+loss.detach().data.cpu().numpy()/batch_size) / (i+1)
                s_mloss=(s_mloss*i+loss2.detach().data.cpu().numpy()/batch_size) / (i+1)
                mloss = (mloss * i + loss1.detach()) / (i + 1)  # update mean losses  loss_items有四个值，box，obj，cls和total
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 4 + '%10.5f'*2)  % (
                    '%g/%g' % (epoch, epochs), mem, a_mloss,s_mloss,mloss[0],targets.shape[0],torch.mean(image_optim).item(),
                                                            torch.mean(image_optim.grad).item()) #这里打印的就是Epoch gpu_mem那一行，targets.shape[0（打印的labels）]显示的就是每一个batch中训练labels总数
                pbar.set_description(s)
            #update texture_para
            # with torch.no_grad():  # 不计算梯度
            #     image_optim.data.clamp_(0.0, 1.0)
            image_optim_in = mix_image(image_optim, mask_image, image_origin)
            if rank in [-1, 0]:
                if i%10==0:
                    Image.fromarray(
                                (255 * image_optim_in).data.cpu().numpy().astype('uint8')).save(
                                os.path.join(log_dir, 'uv贴图in.png'))
                    Image.fromarray(
                                (255 * image_origin).data.cpu().numpy().astype('uint8')).save(
                                os.path.join(log_dir, 'uv贴图out.png'))
                    flip_image(os.path.join(log_dir, 'uv贴图in.png'),os.path.join(log_dir, 'uv贴图in_change_direction.png'))

                    Image.fromarray(
                                (255 * mask_image).data.cpu().numpy().astype('uint8')).save(
                                os.path.join(log_dir, 'mask.png'))
            
            textures=LoadTextures.apply(image_optim_in, faces_3_2, texture_origin, is_update,wrap_way,bilinear_way)
            textures=textures.unsqueeze(0)
            # print(f"textures.shape_render_before:{textures.shape}")
            dataset.set_textures(textures)
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
        if rank in [-1, 0]: 
            tb_writer.add_scalar("meanTLoss", mloss[0], epoch)
            tb_writer.add_scalar("meanSLoss", s_mloss, epoch)
            tb_writer.add_scalar("AllSLoss",a_mloss, epoch)
            if epoch % 1 == 0:
                if epoch % 1 == 0:
                    image_optim_in = mix_image(image_optim, mask_image, image_origin)
                    Image.fromarray(
                            (255 * image_optim_in).data.cpu().numpy().astype('uint8')).save(
                            os.path.join(log_dir, f'texture_{epoch}.png'))
            np.save(os.path.join(log_dir, f'texture_{epoch}.npy'), image_optim_in.data.cpu().numpy())
            flip_image(os.path.join(log_dir, f'texture_{epoch}.png'),os.path.join(log_dir, f'texture_{epoch}.png'))
    torch.cuda.empty_cache()
    return results

log_dir = ""
def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    # dir_name = 'logs_FCA_no_car_paint_withROA_5_5/' + dir_name
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name



if __name__ == '__main__':
    print(f"logger{logger}")
    parser = argparse.ArgumentParser()
    # hyperparameter for training adversarial camouflage
    # ------------------------------------#
    parser.add_argument('--weights', type=str, default='yolov3_9_5.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/carla.yaml', help='data.yaml path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for texture_param')
    parser.add_argument('--obj_file', type=str, default='car_asset_E2E/pytorch3d_Etron.obj', help='3d car model obj') #3D车模
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt',
                        help='exterior_face file  (exterior_face, all_faces)')
    parser.add_argument('--datapath', type=str, default='/home/zjw/data/car_train_total_no_paper/adversarialtrain',
                        help='data path')
    parser.add_argument('--patchInitial', type=str, default='random',
                        help='data path')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lamb", type=float, default=1e-4) #lambda
    parser.add_argument("--d1", type=float, default=0.9)
    parser.add_argument("--d2", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    
    # ------------------------------------#

    #add
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify') #多GPU模型自动修改，不用手动修改
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--classes', nargs='+', type=int, default=[2],
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--continueFrom', type=int, default=0, help='continue from which epoch')
    parser.add_argument('--texturesize', type=int, default=6, help='continue from which epoch')
    opt = parser.parse_args()

    T = opt.t #这个T是计算平滑损失最后乘的参数，论文中的μ
    D1 = opt.d1
    D2 = opt.d2
    lamb = opt.lamb
    LR = opt.lr
    Dataset=opt.datapath.split('/')[-1]
    PatchInitial=opt.patchInitial
    logs = {
        'epoch': opt.epochs,
        'texturesize':opt.texturesize,
        'dataset':Dataset,
        'patchInitialWay':PatchInitial,
        'batch_size': opt.batch_size,
        'T': T, 
        'name': opt.name,
        'no_car_paint':True, 
        'E2E_totalUV':True,
        'UV_map': "2048",
    }

    
    make_log_dir(logs)

    texture_dir_name = ''
    for key, value in logs.items():
        texture_dir_name+= f"{key}-{str(value)}+"
    
    # Set DDP variables
    

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # os.environ[""]获得一个环境变量，world_size是指的是分布式训练使用的进程数或用gpus数
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1 #global_rank表示进程编号，RANK变量不存在则是-1
    print('WORLD_SIZE' in os.environ)
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()   #查当前代码所在的 Git 仓库的状态
        # check_requirements(exclude=('pycocotools', 'thop'))

    



    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run   ``
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))
    opt.total_batch_size=opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    print(f"device:{device}")
    if opt.local_rank != -1:
        msg = 'is not compatible with YOLOv3 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    # Train
    logger.info(opt)
    
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device)



