import argparse
import logging
import os
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw 
from models.yolo import Model
from utils.datasets_NRP_fastest import create_dataloader
from utils.datasets_NRP_test_fastest import create_dataloader as create_dataloader_test
from utils.general_NRP import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, colorstr
from utils.google_utils import attempt_download
from utils.loss_NRP import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from PIL import Image
from Image_Segmentation.network import U_Net
import torch.multiprocessing as mp
import torch.distributed as dist
from skimage.metrics import structural_similarity as SSIM
from itertools import chain
logger = None

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):
    # 
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)# torch.nn.Tanh()()，-11，0.50-1
    return texture_origin * (1 - texture_mask) + texture_mask * textures  #1，，

def calculate_inverse_ratio(masks):
    # 
    nonzero_counts = torch.sum(masks != 0, dim=(1, 2), dtype=torch.float)

    # 
    total_pixels = masks.size(1) * masks.size(2)

    # 
    inverse_ratios = total_pixels / nonzero_counts

    return inverse_ratios
def ssim_metric(target: object, prediction: object, win_size: int=11):
    """
    introduce:
        calculate ssim.
        
    args:
        :param ndarray target: target, like ndarray[256, 256].
        :param ndarray prediction: prediction, like ndarray[256, 256].
        :param int win_size: default.
    
    return:
        :param float cur_ssim: return ssim, between [-1, 1], like 0.72.
    """
    # print(f"target:{target.shape}")
    # print(f"prediction:{prediction.shape}")
    # target_in=target.detach().cpu().numpy().squeeze(0)
    # prediction_in=prediction.detach().cpu().numpy().squeeze(0)
    
    target_in=target.detach().cpu().numpy()
    prediction_in=prediction.detach().cpu().numpy()
    cur_ssim = SSIM(
        target_in,
        prediction_in,
        win_size=win_size,
        data_range=1,
        channel_axis=0
    )

    return cur_ssim
def train(device,hyp, opt,log_dir,logger,):
    
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size,batch_size_test,total_batch_size_test, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size,opt.batch_size_test, opt.total_batch_size_test,opt.weights, opt.nr*opt.gpus+device
    torch.manual_seed(100)
    dist.init_process_group(backend='nccl',init_method='env://',world_size=opt.world_size,rank=rank)
    torch.cuda.set_device(device)
    device=torch.device(device)
    print(f"device:{torch.cuda.current_device()}")

    tb_writer = None  # init loggers
    if rank in [-1, 0]:
        print("")
        prefix = colorstr('tensorboard: ')
        text=f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/"
        print(text)
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    
    train_list=[]
    test_list=[]
    # train_mask_list=[]
    # test_mask_list=[]
    # train_neural_renderer_result_list=[]
    # test_neural_renderer_result_list=[]
    # train_differentColor_list=[]
    # test_differentColor_list=[]
    for datatype in opt.datatype:
        dataTypeDir=os.path.join(opt.datapath,datatype)
        dataTypeDir_test=os.path.join(opt.datapath_test,datatype)
        datatrain=os.path.join(dataTypeDir,'NRPtrain')
        datatest=os.path.join(dataTypeDir_test,'NRPtest')
        train_path=os.path.join(datatrain,'train_new')
        test_path=os.path.join(datatest,'train_new')
        # train_mask=os.path.join(datatrain,'masks')
        # test_mask=os.path.join(datatest,'masks')
        # train_neural_renderer_result=os.path.join(datatrain,'neural_renderer_result')
        # test_neural_renderer_result=os.path.join(datatest,'neural_renderer_result')
        # train_differentColor=os.path.join(datatrain,'differentColor')
        # test_differentColor=os.path.join(datatest,'differentColor')
        train_list.append(train_path)
        test_list.append(test_path)
        # train_mask_list.append(train_mask)
        # test_mask_list.append(test_mask)
        # train_neural_renderer_result_list.append(train_neural_renderer_result)
        # test_neural_renderer_result_list.append(test_neural_renderer_result)
        # train_differentColor_list.append(train_differentColor)
        # test_differentColor_list.append(test_differentColor)
    # ---------------------------------#
    # mask_dir = os.path.join(opt.datapath, 'masks/')
    # mask_dirtest = os.path.join(opt.datapathtest, 'masks/')
    # ---------------------------------#
    # -------Yolo-v3 setting-----------#
    # ---------------------------------#
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)  #sort_keys，false，hyphyp.yaml
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict #data_dictcarla。yaml
    with open(opt.datatest) as f:
        data_dicttest = yaml.safe_load(f)

    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
    #
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    with torch_distributed_zero_first(rank):#
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location=device) 
    # with torch_distributed_zero_first(rank):
    #     check_dataset(data_dict)  # check # load checkpoint
    train_path = data_dict['train']
    test_path = data_dicttest['train']
    gs=32
    imgsz=640
    # print(f"train_path:{train_path}")
    # print(f"test_path:{test_path}")
 
    dataloader, dataset,sampler = create_dataloader(train_list, imgsz,device, batch_size, gs,  opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers_train,
                                            prefix=colorstr('train: '), ret_mask=True)##mask
    # ---------------------------------#
    # -------Yolo-v3 setting-----------#
    # ---------------------------------#
    #dataset.start()
    print(f"test_path:{test_path}")
    
    nb = len(dataloader)  # number of batches
    print(f"nb:{nb}")
    


    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    model_NRP=U_Net()
    if opt.zhuanyi:
        saved_state_dict = torch.load("NRP_weights_no_car_paint/generalized_object_EFE_weight/model_NRP_l51.pth")  # 
        new_state_dict = {}
        for k, v in saved_state_dict.items():
            name = k[7:]  #  'module.' 
            new_state_dict[name] = v
        saved_state_dict = new_state_dict
        model_NRP.load_state_dict(saved_state_dict)

    model_NRP.to(device)
    model_NRP = nn.SyncBatchNorm.convert_sync_batchnorm(model_NRP)
    # model_NRP.to(device)
    # model_NRP.cuda(device)
    model_NRP=nn.parallel.DistributedDataParallel(model_NRP,device_ids=[device],output_device=device)
    # model_fully=fully_connected()
    # model_fully.to(device)
    # model_fully = nn.SyncBatchNorm.convert_sync_batchnorm(model_fully)
    # model_fully=nn.parallel.DistributedDataParallel(model_fully,device_ids=[device],output_device=device)
    # if rank in [-1, 0]:
    #     init_img =torch.zeros((1,3,640,640),device=device)
    #     tb_writer.add_graph(model_NRP, init_img)
    # parameters = chain(model_NRP.parameters(), model_fully.parameters())
    num_parameters = sum(p.numel() for p in model_NRP.parameters())
    print(f"Number of parameters: {num_parameters}")
    # optimizer = optim.Adam(model_NRP.parameters(), lr=0.001,weight_decay=1e-5)
    optimizer = optim.Adam(model_NRP.parameters(), lr=0.01)
    optimizer.zero_grad()
    # ---------------------------------#
    # ------------Training-------------#
    # ---------------------------------#
    epoch_start=1+opt.continueFrom
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    number_total=opt.epochs
    for epoch_real in range(1,number_total+1):
    # for epoch_real in range(0,1):
        if epoch_real!=0:
            sampler.set_epoch(epoch_real)
            mloss = torch.zeros(1, device=device)
            mloss_same = torch.zeros(1, device=device)
            if epoch_real==0:
                model_NRP.eval()
            else:
                model_NRP.train()
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem','mloss','loss'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            #， texture_param
            #print(dataloader)
            record_start_all=time.perf_counter()
            # dataset.set_color(epoch)
            # print(f"GPU0:{torch.cuda.memory_reserved() / 1E9}")
            for i, (imgs, texture_img_list, masks,imgs_cut,imgs_NRP_ref_list, targets, paths, _) in pbar:  # batch ------------------------------------------------------------
                #
                # print(f"GPU1:{torch.cuda.memory_reserved() / 1E9}")
                imgs_cut = imgs_cut.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                imgs_in= imgs_cut[0]*masks[0]+imgs[0]*(1-masks[0])/ 255.0 
                
                out_tensor = model_NRP(imgs_cut)  # forward
                sig = nn.Sigmoid()
                out_tensor=sig(out_tensor)
                tensor1 = out_tensor[:,0:3, :, :]
                tensor2 = out_tensor[:,3:6, :, :]


                texture_img_list = texture_img_list.transpose(0, 1)
                # imgs_NRP_ref_list = imgs_NRP_ref_list.transpose(0, 1)
                tensor1=tensor1.unsqueeze(0)
                tensor2=tensor2.unsqueeze(0)
                tensor3=texture_img_list*tensor1+tensor2
                tensor3 = tensor3.transpose(1, 0)
                q=(i+epoch_real)%opt.train_color
                
                
                loss_array_MSE= torch.zeros(tensor3.shape[0]).to(device)
                for p, (texture_img, imgs_NRP_ref) in enumerate(zip(tensor3, imgs_NRP_ref_list)):
                    loss_array_MSE[p]=criterion(texture_img,imgs_NRP_ref)
                ratio = calculate_inverse_ratio(masks)
                loss=torch.sum(loss_array_MSE*ratio)
                output=torch.clamp(tensor3[0][q],0,1)*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                #output=tensor3[0][q]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                output_ref=imgs_NRP_ref_list[0][q]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                # Backward
                optimizer.zero_grad()
                loss.backward(retain_graph=False) #retain_graph=True ，，texture_param.grad
                if epoch_real!=0:
                    optimizer.step()    
                loss_copy = loss.clone().detach()
                dist.all_reduce(loss_copy)
                dist.barrier()
                
                                        
                # try:
                #     if rank in [-1, 0]: 
                #         if i % 40 == 0:
                #     # Image.fromarrayarray
                #     # imgs.data.cpu().numpy()[0]imgs，255，np.transpose(..., (1, 2, 0))： (C, H, W)  (H, W, C)，（ RGB）。
                #     #     print(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8').shape)
                #     #     print(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8'))
                #     #     Image.fromarray(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                #     #         os.path.join(log_dir, '.png')) 
                #     #     print(os.path.join(log_dir, '.png'))
                #     # # image_ref.save(os.path.join(log_dir, 'color_ref.png'))
                #             Image.fromarray((255 * masks).data.cpu().numpy()[0].astype('uint8')).save(
                #                 os.path.join(log_dir, '.png'))
                #             Image.fromarray(
                #                 (255 * texture_img_list[q][0]).data.cpu().numpy().transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, '.png')) #，
                            
                            
                #             Image.fromarray(
                #                 (255 * tensor1).data.cpu().numpy()[0][0].transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, 'img_tensor1.png'))
                #             Image.fromarray(
                #                 (255 * tensor2).data.cpu().numpy()[0][0].transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, 'img_tensor2.png'))
                #         # #     Image.fromarray(
                #         #         (255 * tensor3).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                #         #         os.path.join(log_dir, 'result.png'))
                #             Image.fromarray(
                #                 (255 * imgs_NRP_ref_list[0][q]).data.cpu().numpy().transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, 'target.png'))
                #             Image.fromarray(
                #                     (255*imgs_cut).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                #                     os.path.join(log_dir, ".png"))
                            
                #             Image.fromarray(np.transpose(255*imgs_in.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, '.png')) 

                #             Image.fromarray(np.transpose(255*output.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, '.png')) 
                            
                #             Image.fromarray(np.transpose(255*output_ref.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir, 'ref.png'))
                    
                # except:
                #     pass
                
                # print("model_NRP.conv1.weight.data:",model_NRP.module.Conv_1x1.weight.data.mean())
                #print(f"i:{i} color:{color_name}")
                
                if rank in [-1, 0]: 
                    # j=i*total_batch_size+9*nb*total_batch_size
                    print(f"total_batch_size{total_batch_size}")
                    mloss = (mloss*i*total_batch_size  + loss_copy) / (i*total_batch_size + total_batch_size) 
                    
                    # update mean losses  loss_items，box，obj，clstotal
                    # mloss_same = (mloss_same *j  + loss_copy) / (j + total_batch_size)  # update mean losses  loss_items，box，obj，clstotal
                    # print(total_batch_size)
                    # print(loss_copy.data)
                    # print(loss.data)

                        
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 +"%10.4f"*2)  % (
                        '%g/%g' % (epoch_real, number_total), mem,mloss.data,loss_copy.data) #Epoch gpu_mem，targets.shape[0（labels）]batchlabels
                        #，mlossloss.data.cpu()，get_item()0.2，SimpleNRP0.1，，get_item()0.3，SimpleNRP0.0，？
                    pbar.set_description(s)
            if rank in [-1, 0]: 
                torch.save(model_NRP.state_dict(), (os.path.join(log_dir, f'model_NRP_l{epoch_real}.pth')))
                # torch.save(model_fully.state_dict(), (os.path.join(log_dir, f'model_fully_s{epoch}_l{epoch_real}.pth')))
                tb_writer.add_scalar(f"TrainLoss_MSEMeanloss_{opt.datatype}_no_car_paint", mloss.data, epoch_real)
    
    torch.cuda.empty_cache()
    dataloader_test, dataset_test,sampler_test = create_dataloader_test(test_list, 640, device,batch_size_test, 32,  opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers_test,
                                            prefix=colorstr('train: '), ret_mask=True)
    nb_test = len(dataloader_test)  # number of batches
    number=number_total
    
    for epoch_real in range(0,number+1):
        sampler_test.set_epoch(epoch_real)
        epoch_test=epoch_real
        model_NRP=None
        model_NRP=U_Net()
        model_NRP.to(device)
        if opt.zhuanyi:
            if epoch_test==0:
                saved_state_dict = torch.load("NRP_weights_no_car_paint/generalized_object_EFE_weight/model_NRP_l51.pth")  # 
                new_state_dict = {}
                for k, v in saved_state_dict.items():
                    name = k[7:]  #  'module.' 
                    new_state_dict[name] = v
                saved_state_dict = new_state_dict
                model_NRP.load_state_dict(saved_state_dict)
        if epoch_test!=0:
            path=os.path.join(log_dir,f"model_NRP_l{epoch_test}.pth")
            saved_state_dict = torch.load(path)  # 
            new_state_dict = {}
            for k, v in saved_state_dict.items():
                name = k[7:]  #  'module.' 
                new_state_dict[name] = v
            saved_state_dict = new_state_dict
            model_NRP.load_state_dict(saved_state_dict)
        with torch.no_grad():
            BCE = nn.BCELoss()
            MSE=nn.MSELoss()
            MAE=nn.L1Loss()
            ssim =ssim_metric
            
            model_NRP.eval()
            # model_fully.eval()
            mlossBCE=0
            mlossMSE=0
            mlossMAE=0
            mlossSSIM=0
            mlossBCE_no_ratio =0
            mlossMSE_no_ratio =0
            mlossMAE_no_ratio =0
            # for epoch in range(1, 64+1):
                
            pbar = enumerate(dataloader_test)
            # textures = cal_texture(texture_param, texture_origin, texture_mask) #，texture_mask
            # dataset.set_textures(textures) #，，texture_img 。masks，texture_img
            # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'loss','labels','tex_mean','grad_mean'))
            logger.info(('\n' + '%10s' * 10) % ('Epoch', 'gpu_mem','BCEmloss','BCEloss','MSEmloss','MSEloss','MAEmloss','MAEloss','SSIMmloss','SSIMloss'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb_test)  # progress bar
            #， texture_param
            #print(dataloader)
            # print(f"epoch{epoch}")
            # dataset_test.set_color(epoch)
            # end_first=time.perf_counter()
            for i, (imgs, texture_img_list, masks,imgs_cut,imgs_NRP_ref_list, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                
                # start_first=time.perf_counter()
                # print(f"dataloader:{end_first-start_first}")
                
                #
                imgs_cut = imgs_cut.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
                imgs_in= imgs_cut[0]*masks[0]+imgs[0]*(1-masks[0])/ 255.0 
                
                out_tensor = model_NRP(imgs_cut)  # forward
                sig = nn.Sigmoid()
                out_tensor=sig(out_tensor)
                tensor1 = out_tensor[:,0:3, :, :]
                tensor2 = out_tensor[:,3:6, :, :]
                #compute loss
                texture_img_list = texture_img_list.transpose(0, 1)
                # imgs_NRP_ref_list = imgs_NRP_ref_list.transpose(0, 1)
                tensor1=tensor1.unsqueeze(0)
                tensor2=tensor2.unsqueeze(0)
                tensor3=torch.clamp(texture_img_list*tensor1+tensor2,0,1)
                tensor3 = tensor3.transpose(1, 0)
                #tensor3shape2，32，3，640，640
                masks_enlarge=masks.unsqueeze(1).unsqueeze(2).repeat(1,tensor3.shape[1],tensor3.shape[2],1,1)
                tensor3=tensor3*masks_enlarge

                #
                # if not os.path.exists(os.path.join(log_dir,"tensor3")):
                #     os.mkdir(os.path.join(log_dir,"tensor3"))
                # if not os.path.exists(os.path.join(log_dir,"ref")):
                #     os.mkdir(os.path.join(log_dir,"ref"))
                # for y in range(tensor3.shape[0]):
                #     for x in range(tensor3.shape[1]):
                #         Image.fromarray(
                #                 (255 * tensor3[y][x]).data.cpu().numpy().transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir,"tensor3",f'{y}_{x}.png'))
                #         Image.fromarray(
                #                 (255 * imgs_NRP_ref_list[y][x]).data.cpu().numpy().transpose((1, 2, 0)).astype('uint8')).save(
                #                 os.path.join(log_dir,"ref",f'{y}_{x}.png'))
                q=(i+epoch_real)%opt.test_color
                output=tensor3[0][q]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                output_ref=imgs_NRP_ref_list[0][q]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])

                loss_array_BCE= torch.zeros(tensor3.shape[0]).to(device)
                loss_array_MSE= torch.zeros(tensor3.shape[0]).to(device)
                loss_array_MAE= torch.zeros(tensor3.shape[0]).to(device)
                for p, (texture_img, imgs_NRP_ref) in enumerate(zip(tensor3, imgs_NRP_ref_list)):
                    loss_array_BCE[p]=BCE(texture_img,imgs_NRP_ref)
                    loss_array_MSE[p]=MSE(texture_img,imgs_NRP_ref)
                    loss_array_MAE[p]=MAE(texture_img,imgs_NRP_ref)
                ratio = calculate_inverse_ratio(masks)
                loss_BCE=torch.sum(loss_array_BCE*ratio).data
                loss_MSE=torch.sum(loss_array_MSE*ratio).data
                loss_MAE=torch.sum(loss_array_MAE*ratio).data
                loss_SSIM=torch.zeros(1).to(device)

                # ratioloss
                loss_BCE_no_ratio=BCE(tensor3,imgs_NRP_ref_list).data
                loss_MSE_no_ratio=MSE(tensor3,imgs_NRP_ref_list).data
                loss_MAE_no_ratio=MAE(tensor3,imgs_NRP_ref_list).data
                
                # loss_SSIM=torch.from_numpy(np.asarray(ssim(tensor3,imgs_NRP_ref_list)))
                dist.all_reduce(loss_BCE)
                dist.all_reduce(loss_MSE)
                dist.all_reduce(loss_MAE)
                dist.all_reduce(loss_SSIM)
                dist.all_reduce(loss_BCE_no_ratio)
                dist.all_reduce(loss_MSE_no_ratio)
                dist.all_reduce(loss_MAE_no_ratio)
                dist.barrier()
                
                                            
                try:
                    if rank in [-1, 0]: 
                        if i % 40 == 0:
                            #Image.fromarrayarray
                            #imgs.data.cpu().numpy()[0]imgs，255，np.transpose(..., (1, 2, 0))： (C, H, W)  (H, W, C)，（ RGB）。
                                # print(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8').shape)
                                # print(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8'))
                                # Image.fromarray(np.transpose(imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                                #     os.path.join(log_dir, '.png')) 
                                # print(os.path.join(log_dir, '.png'))
                            # # image_ref.save(os.path.join(log_dir, 'color_ref.png'))
                            # Image.fromarray((255 * masks).data.cpu().numpy()[0].astype('uint8')).save(
                            #     os.path.join(log_dir, '.png'))
                            Image.fromarray(
                                (255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                                os.path.join(log_dir, '_eval.png')) #，
                            Image.fromarray(np.transpose(255*imgs_in.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                                    os.path.join(log_dir, '_val.png')) 
                            Image.fromarray(
                                (255 * tensor1).data.cpu().numpy()[0][0].transpose((1, 2, 0)).astype('uint8')).save(
                                os.path.join(log_dir, 'img_tensor1_val.png'))
                            Image.fromarray(
                                (255 * tensor2).data.cpu().numpy()[0][0].transpose((1, 2, 0)).astype('uint8')).save(
                                os.path.join(log_dir, 'img_tensor2_val.png'))
                            # Image.fromarray(
                            #     (255 * tensor3).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            #     os.path.join(log_dir, 'result.png'))
                            # Image.fromarray(
                            #     (255 * imgs_NRP_ref).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            #     os.path.join(log_dir, 'target.png'))
                            # Image.fromarray(
                            #         (255*imgs_cut).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            #         os.path.join(log_dir, ".png"))
                            Image.fromarray(np.transpose(255*output.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                                os.path.join(log_dir, '_val.png')) 
                            
                            Image.fromarray(np.transpose(255*output_ref.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                                os.path.join(log_dir, 'ref_val.png'))
                
                except:
                    pass
                
                if rank in [-1, 0]: 
                    j=i*opt.total_batch_size_test#+(epoch-1)*nb_test*total_batch_size_test
                    mlossBCE = (mlossBCE *j  + loss_BCE) / (j + opt.total_batch_size_test)  # update mean losses  loss_items，box，obj，clstotal
                    mlossMSE = (mlossMSE *j  + loss_MSE) / (j + opt.total_batch_size_test)  # update mean losses  loss_items，box，obj，clstotal
                    mlossMAE = (mlossMAE *j  + loss_MAE) / (j + opt.total_batch_size_test)  # update mean losses  loss_items，box，obj，clstotal
                    mlossSSIM = (mlossSSIM *j  + loss_SSIM) / (j + opt.total_batch_size_test)


                    k=i*opt.gpus
                    mlossBCE_no_ratio = (mlossBCE_no_ratio *k  + loss_BCE_no_ratio) / (k + opt.gpus)  # update mean losses  loss_items，box，obj，clstotal
                    mlossMSE_no_ratio = (mlossMSE_no_ratio *k  + loss_MSE_no_ratio) / (k + opt.gpus)  # update mean losses  loss_items，box，obj，clstotal
                    mlossMAE_no_ratio = (mlossMAE_no_ratio *k  + loss_MAE_no_ratio) / (k + opt.gpus)  # update mean losses  loss_items，box，obj，clstotal
                    print(mlossMAE_no_ratio)
                    # update mean losses  loss_items，box，obj，clstotal
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 +"%10.4f"*8)  % (
                        '%g/%g' % (epoch_real,number_total), mem,mlossBCE,loss_BCE,mlossMSE,loss_MSE,mlossMAE,loss_MAE,mlossSSIM,loss_SSIM) #Epoch gpu_mem，targets.shape[0（labels）]batchlabels
                    #，mlossloss.data.cpu()，get_item()0.2，SimpleNRP0.1，，get_item()0.3，SimpleNRP0.0，？
                    pbar.set_description(s)
            if rank in [-1, 0]:  
                tb_writer.add_scalar(f"Test_BCEMeanloss_{opt.datatype}_no_car_paint", mlossBCE, epoch_test)
                tb_writer.add_scalar(f"Test_MSEMeanloss_{opt.datatype}_no_car_paint", mlossMSE, epoch_test)
                tb_writer.add_scalar(f"Test_MAEMeanloss_{opt.datatype}_no_car_paint", mlossMAE, epoch_test)
                tb_writer.add_scalar(f"Test_SSIMMeanloss_{opt.datatype}_no_car_paint", mlossSSIM, epoch_test)

                tb_writer.add_scalar(f"Test_BCEMeanloss_no_ratio_{opt.datatype}_no_car_paint", mlossBCE_no_ratio, epoch_test)
                tb_writer.add_scalar(f"Test_MSEMeanloss_no_ratio_{opt.datatype}_no_car_paint", mlossMSE_no_ratio, epoch_test)
                tb_writer.add_scalar(f"Test_MAEMeanloss_no_ratio_{opt.datatype}_no_car_paint", mlossMAE_no_ratio, epoch_test)
                print(f"model_number:{epoch_test},BCEloss:{mlossBCE},MSEloss:{mlossMSE},MAEloss:{mlossMAE},SSIMloss:{mlossSSIM}")
        
            
    print(f"nb_test:{nb_test}")
    if rank in [-1, 0]: 
        torch.save(model_NRP.state_dict(), 'model_NRP.pth')
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results

log_dir = ""
def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs_NRP/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name



if __name__ == '__main__':
    logger= logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    # hyperparameter for training adversarial camouflage
    # ------------------------------------#
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/NRP_boat.yaml', help='data.yaml path')
    parser.add_argument('--datatest', type=str, default='data/NRP_boat.yaml', help='data.yaml path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for texture_param')
    parser.add_argument('--obj_file', type=str, default='car_assets/audi_et_te.obj', help='3d car model obj') #3D
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt',
                        help='exterior_face file  (exterior_face, all_faces)')
    parser.add_argument('--datapath', type=str, default='/CarlaDataset_test_6_1_specific_16viewpoint_train',
                        help='data path')
    parser.add_argument('--datapath_test', type=str, default='/CarlaDataset_test_6_1_specific_16viewpoint_train',
                        help='data path')
    parser.add_argument
    # parser.add_argument('--datatype', nargs='+',type=str, default=["car_simple", "cube","boat_simple","sphere","cylinder"],
    #                     help='data path')
    parser.add_argument('--datatype', nargs='+',type=str, default=["car", "cube","boat_simple","sphere","cylinder","benz","citreon"],
                        help='data path')
    parser.add_argument('--patchInitial', type=str, default='random',
                        help='data path')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lamb", type=float, default=1e-4) #lambda
    parser.add_argument("--d1", type=float, default=0.9)
    parser.add_argument("--d2", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=320)
    parser.add_argument('--train_color', type=int, default=9)
    parser.add_argument('--test_color', type=int, default=32)
    parser.add_argument('--zhuanyi', action='store_true')

    # ------------------------------------#

    #add
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--batch-size-test', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers_train', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--workers_test', type=int, default=10, help='maximum number of dataloader workers')
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
    parser.add_argument('--nodes',type=int,default=1)
    parser.add_argument('--gpus',type=int,default=4,help="num gpus per node")
    parser.add_argument('--nr',type=int,default=0,help="ranking within the nodes")


    opt = parser.parse_args()

    T = opt.t #T，μ
    D1 = opt.d1
    D2 = opt.d2
    lamb = opt.lamb
    LR = opt.lr
    Dataset=opt.datapath.split('/')[-1]
    # Datasettest=opt.datapathtest.split('/')[-1]
    PatchInitial=opt.patchInitial
    logs = {
        'epoch-real':opt.epochs,
        'dataset_train':Dataset,
        'dataset_type':opt.datatype,
        "fastest" : True,
        #'pretrainmodel' :"13",
        "name" : opt.name,
    }
    
    make_log_dir(logs)
    print(logs)
    texture_dir_name = ''
    for key, value in logs.items():
        texture_dir_name+= f"{key}-{str(value)}+"
    
    # Set DDP variables


    # opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # os.environ[""]，world_size
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # set_logging(opt.global_rank)
    # print(f"rank:{opt.global_rank}")
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()
    #     check_requirements(exclude=('pycocotools', 'thop'))

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

    opt.total_batch_size = opt.batch_size
    opt.total_batch_size_test = opt.batch_size_test
    # device = select_device(opt.device, batch_size=opt.batch_size)

    # #add
    # if opt.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    opt.world_size=opt.nodes*opt.gpus
    opt.lr=opt.lr*opt.world_size
    opt.batch_size = opt.total_batch_size // opt.world_size
    opt.batch_size_test = opt.total_batch_size_test // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    # Train
    logger.info(opt)
    
    # mlossBCE = torch.zeros(1)
    # mlossMSE = torch.zeros(1)
    # mlossMAE = torch.zeros(1)
    # mlossSSIM = torch.zeros(1)
    # count = torch.zeros(1)
    # mlossBCE.share_memory_()
    # mlossMSE.share_memory_()
    # mlossMAE.share_memory_()
    # mlossSSIM.share_memory_()
    # count.share_memory_()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1121'
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.device
    mp.spawn(train,nprocs=opt.gpus,args=(hyp,opt,log_dir,logger,),join=True)
    



