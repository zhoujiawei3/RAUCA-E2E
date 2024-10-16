# Dataset utils and dataloaders

import glob
import hashlib
import logging
import os
import random
import shutil
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import math
from utils.general import xywhn2xyxy, segments2boxes,  xyxy2xywhn
from utils.torch_utils import torch_distributed_zero_first
import utils.nmr_test_boat as nmr

# Parameters
help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz,device, batch_size, stride,opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='',mask_dir='',ret_mask=False, phase='training'):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, device,imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular trainingF
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix, mask_dir=mask_dir, ret_mask=ret_mask, phase=phase)

    batch_size = min(batch_size, len(dataset))
    # nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers os.cpu_count()CPU，world_size。
    nw = min([os.cpu_count() // world_size, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=rank) if rank != -1 else None #rank=-1，
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader #
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset,sampler


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    #


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def img2label_paths(img_paths, phase='training'):
    # Define label paths as a function of image paths
    if phase == 'training':
        sa, sb = os.sep + 'train_new' + os.sep, os.sep + 'train_label_new' + os.sep  # /images/, /labels/ substrings
    else:
        sa, sb = os.sep + 'test_new' + os.sep, os.sep + 'test_label_new' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, device, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',mask_dir='', ret_mask=False,phase='training'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.phase = phase
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]: #
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True) #pf
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats]) #，/，
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files, phase)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        # cache.pop('hash')  # remove hash
        # cache.pop('version')  # remove version
        if phase == 'training':
            [cache.pop(k) for k in ('hash', 'version')]  # remove items
            # [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        else:
            [cache.pop(k) for k in ('hash', 'version')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)  #
        self.shapes = np.array(shapes, dtype=np.float64) #turple，，[800,800]，[[800,800],...]
        # print(f"shapes:{self.shapes}")
        self.img_files = list(cache.keys())  # update  ['/data/zhoujw/upload_NSR/train_new/data10.png', '/data/zhoujw/upload_NSR/train_new/data100.png', '/data/zhoujw/upload_NSR/train_new/data1000.png'...]
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index  [0,1,2,3,4,5..][0,0,0,1,1,1,2,2,2..]batch_size3
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio ,
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i], _ = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()
        # renderer
        self.device = device
        
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        # R=['0','85','170','255']
        # G=[',0',',85',',170',',255']
        # B=[',0',',85',',170',',255']

        # self.color_list=[R[x%4]+G[x//4%4]+B[x//16%4] for x in range(64)]
        parent_dir = os.path.dirname(path[0])
        new_path = os.path.join(parent_dir, "diffferentColor")
        names = os.listdir(new_path)
        self.color_list =[name for name in names if os.path.isdir(os.path.join(new_path, name))]
        
        # print(self.color_list)
    # @property
    # def mask_renderer(self):
    #     mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).to(self.device)
    #     mask_renderer.renderer.renderer.camera_mode = "look_at"
    #     mask_renderer.renderer.renderer.light_direction = [0, 0, 1]
    #     mask_renderer.renderer.renderer.camera_up = [0, 0, 1]
    #     mask_renderer.renderer.renderer.background_color = [1, 1, 1]
    #     return mask_renderer

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                logging.info(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.2  # cache version
        try:
            torch.save(x, path)  # save cache for next time
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def set_textures(self, textures):
        self.textures = textures
    def set_epoch(self,epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.img_files)
    # def set_color(self,color_add):
    #     # print("")
    #     self.color_list_add=color_add

    def set_textures_255(self, textures_255):
        self.textures_255=textures_255

    def __getitem__(self, index):
        with torch.no_grad():
            
            if self.ret_mask:
                mask_dir=os.path.join(os.path.dirname(os.path.dirname(self.img_files[index])),'masks')
                mask_file = os.path.join(mask_dir, "%s.png" % os.path.basename(self.img_files[index])[:-4])
                #print(f"maskfile{mask_file}")
                mask = cv2.imread(mask_file)
                mask = cv2.resize(mask, (self.img_size, self.img_size))

                mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]) #
                mask = torch.from_numpy(mask.astype('float32')).to(self.device)
        
            img, (h0, w0), (h, w), (veh_trans, cam_trans),path = load_image(self, index) #
            length=math.sqrt(cam_trans[0][0]**2+
                                            cam_trans[0][1]**2+
                                            cam_trans[0][2]**2)
            
            new_path = os.path.join(os.path.dirname(os.path.dirname(path)),"diffferentColor")
            names = os.listdir(new_path)
            color_list=[name for name in names if os.path.isdir(os.path.join(new_path, name))]
            #camera parameters
            render_image_list=[]
            for i in color_list:
                color_name=i
                neural_img_filename = str(round(length))+"_"+ str(cam_trans[1][0])+"_"+ str(cam_trans[1][1])+"_"+ str(cam_trans[1][2])+"_"+ ".png"
                neural_img_dir=os.path.join(os.path.dirname(os.path.dirname(path)),'neural_renderer_result')
                neural_img_file_path=os.path.join(neural_img_dir,color_name,neural_img_filename)
                
                imgs_pred = cv2.imread(neural_img_file_path)
                imgs_pred =cv2.cvtColor(imgs_pred,cv2.COLOR_BGR2RGB)
                imgs_pred=torch.from_numpy(np.array(imgs_pred)).unsqueeze(0)
                imgs_pred=imgs_pred.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0 non_blocking，
                imgs_pred = imgs_pred.permute(0,3,1,2)
                imgs_pred=mask*imgs_pred
                render_image_list.append(imgs_pred.squeeze(0))
            #
            render_image_list=torch.stack(render_image_list, 0)
            
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape ：640
        
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy() #
            #，label
            

            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            nl = len(labels)  # number of labels
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])  # xyxy to xywh normalized

            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)
            # Convert
            
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img) #
            img = torch.from_numpy(img).to(self.device)
            
            img_cut=( img) * mask
            # img_background=( img) * (1-mask)
            # img_forground=img*mask
            # img_later=img_forground+img_background
            # Image.fromarray(np.transpose(img_later.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
            #                     os.path.join("/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs_new/epoch-9+epoch-real-200+dataset-boat_multi_weather+method-(a*b+c)+tensor-2+ratio-true+boat-true+sync-true+patchInitialWay-random+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+", 'later.png'))
            
            # Image.fromarray(np.transpose(img_forground.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
            #                     os.path.join("/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs_new/epoch-9+epoch-real-200+dataset-boat_multi_weather+method-(a*b+c)+tensor-2+ratio-true+boat-true+sync-true+patchInitialWay-random+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+", 'foreground.png'))
            
            # Image.fromarray(np.transpose(img_background.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
            #                     os.path.join("/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs_new/epoch-9+epoch-real-200+dataset-boat_multi_weather+method-(a*b+c)+tensor-2+ratio-true+boat-true+sync-true+patchInitialWay-random+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+", 'background.png'))
            # # Applying mask, the transformation function in paper
            img = (1 - mask) * img 
            
            filename=self.img_files[index].split('/')[-1]
            color_ref_dir=os.path.join(os.path.dirname(os.path.dirname(self.img_files[index])),'diffferentColor')
            ref_image_list=[]
            for i in color_list:
                color_name=i
                color_ref_path=os.path.join(color_ref_dir,color_name,filename)
                image_ref = cv2.imread(color_ref_path)
                image_ref =cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB)
            # image_ref =Image.open(color_ref_path).convert("RGB")
                image_ref_tensor=torch.from_numpy(np.array(image_ref)).unsqueeze(0)
                image_NSR_ref =image_ref_tensor.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0 non_blocking，
                image_NSR_ref = image_NSR_ref.permute(0,3,1,2)
                image_NSR_ref = image_NSR_ref*mask
                ref_image_list.append(image_NSR_ref.squeeze(0))
            ref_image_list=torch.stack(ref_image_list, 0)
            # color_ref_path=os.path.join(color_ref_dir,color_name,filename)

            
            # # print(color_ref_path)
            # image_ref = cv2.imread(color_ref_path)
            # image_ref =cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB)
            # # image_ref =Image.open(color_ref_path).convert("RGB")
            # image_ref_tensor=torch.from_numpy(np.array(image_ref)).unsqueeze(0)
            # image_NSR_ref =image_ref_tensor.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0 non_blocking，
            # image_NSR_ref = image_NSR_ref.permute(0,3,1,2)
            # image_NSR_ref = image_NSR_ref*mask
            
            # print(img)
            # return img.squeeze(0), imgs_pred.squeeze(0), mask, imgs_ref.squeeze(0),img_cut.squeeze(0),labels_out, self.img_files[index], shapes

            #print(f"render:{end-start},color:{end_color-start_color},mask:{end_mask-start_mask},load_all:{end_load-start_load}")
            # print(f"img.shape{img.shape}")
            # print(f"render_image_list.shape{render_image_list.shape}")
            return img.squeeze(0), render_image_list.squeeze(0), mask,img_cut.squeeze(0),ref_image_list.squeeze(0),labels_out, self.img_files[index], shapes


    @staticmethod
    def collate_fn(batch):
        #img, texture_img, masks,imgs_ref,img_cut, label, path, shapes = zip(*batch)  # transposed
        img, render_image_list, masks,img_cut, ref_image_list,label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        #return torch.stack(img, 0), torch.stack(texture_img, 0),torch.stack(masks, 0),torch.stack(imgs_ref, 0), torch.stack(img_cut, 0),torch.cat(label, 0), path, shapes
        return torch.stack(img, 0), torch.stack(render_image_list, 0),torch.stack(masks, 0), torch.stack(img_cut, 0), torch.stack(ref_image_list, 0),torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


def load_image(self, index):

    """
    Load simulated image and location inforamtion
    """

    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    if self.phase == 'training':
        sa, sb = os.sep + 'train_new' + os.sep, os.sep + 'train' + os.sep  # /images/, /labels/ substrings
    else:
        sa, sb = os.sep + 'test_new' + os.sep, os.sep + 'test' + os.sep  # /images/, /labels/ substrings

    path = sb.join(path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.npz'
    data = np.load(path, allow_pickle=True)  # .item() # .item()      #
    img = data['img']
    # img = img[:, :, ::-1]  # 
    # the relation among veh_trans or cam_trans and img
    veh_trans, cam_trans = data['veh_trans'], data['cam_trans']
    # cam_trans[0][2]-=0.81
    
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                         interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return img, (h0, w0), img.shape[:2], (veh_trans, cam_trans),path  # img, hw_original, hw_resized ，h0，w0，h，wresize


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
