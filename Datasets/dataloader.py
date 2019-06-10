import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read
import pdb


def get_loader(args, dataset):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args.crop_h, args.crop_w)
    perform_transformation = not args.no_aug

    train_dataset = Dataset_loader(
            args.data_path, dataset.train_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=perform_transformation)
    val_dataset = Dataset_loader(
            args.data_path, dataset.val_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False)
    val_select_dataset = Dataset_loader(
            args.data_path, dataset.selected_paths, args.input_type,
            resize=None, rotate=args.rotate, crop=crop_size,
            flip=args.flip, rescale=args.rescale, max_depth=args.max_depth,
            sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False)

    train_sampler = None
    val_sampler = None
    if args.subset is not None:
        random.seed(1)
        train_idx = [i for i in random.sample(range(len(train_dataset)-1), args.subset)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        random.seed(1)
        val_idx = [i for i in random.sample(range(len(val_dataset)-1), round(args.subset*0.5))]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=int(args.val_batch_size),  sampler=val_sampler,
        shuffle=val_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_selection_loader = DataLoader(
        val_select_dataset, batch_size=int(args.val_batch_size), shuffle=False,
        num_workers=args.nworkers, pin_memory=True, drop_last=True)
    return train_loader, val_loader, val_selection_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, data_path, dataset_type, input_type, resize,
                 rotate, crop, flip, rescale, max_depth, sparse_val=0.0, 
                 normal=False, disp=False, train=False):
        self.use_rgb = input_type == 'rgb'
        # self.use_rgb = True
        self.datapath = data_path
        self.dataset_type = dataset_type
        self.train = train
        self.resize = resize
        self.flip = flip
        self.crop = crop
        self.rotate = rotate
        self.rescale = rescale
        # self.bound = 1.15 if self.rescale else 1.0
        self.bound = 1.0

        # if you rescale depth ==> resize the image!
        self.lowerbound = 0.85 if self.rescale else 1.0
        # self.lowerbound = 1.0
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)

        # Imagenet normalization for rgb images
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.depth_norm = transforms.Normalize(mean=[14.97/max_depth], std=[11.15/max_depth])
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.normal = normal
        self.max_depth = max_depth
        self.sparse_val = sparse_val
        self.disp = disp

        self.img_name = 'img'
        self.lidar_name = 'lidar_in' 
        self.gt_name = 'gt' 

    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['lidar_in'])


    def define_transforms(self, input, gt, img=None):

        # Define random variabels
        angle = np.random.uniform(-5.0, 5.0) and self.rotate
        angle = int(angle) if type(angle) == bool else angle
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'
        vflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'vflip'
        self.scale = np.random.uniform(self.lowerbound, self.bound)

        if self.train:
            input, gt = F.rotate(input, angle), F.rotate(gt, angle)
            i, j, h, w = transforms.RandomCrop.get_params(input, output_size=self.crop)
            input = F.crop(input, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)
            if hflip_input:
                input, gt = F.hflip(input), F.hflip(gt)
            elif vflip_input:
                input, gt = F.vflip(input), F.vflip(gt)

            if self.use_rgb:
                img = F.rotate(img, angle)
                # img = misc.imresize(img, self.scale) #'nearest')
                # img = Image.fromarray(img)
                img = F.crop(img, i, j, h, w)
                # img = self.center_crop(img)
                if hflip_input:
                    img = F.hflip(img)
                elif vflip_input:
                    img = F.vflip(img)
                # img = self.color_jitter(img)

            input, gt = depth_read(input, self.sparse_val)/self.scale, depth_read(gt, self.sparse_val)/self.scale
        else:
            # input, gt = F.crop(input, 130, 10, 240, 1200), F.crop(gt, 130, 10, 240, 1200)
            input, gt = self.center_crop(input), self.center_crop(gt)
            if self.use_rgb:
                # img = F.crop(img)

                img = self.center_crop(img)
            input, gt = depth_read(input, self.sparse_val), depth_read(gt, self.sparse_val)


        return input, gt, img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of pred velodyne data and ground truth.
        """
        sparse_depth = os.path.join(self.dataset_type[self.lidar_name][idx])
        gt_name = os.path.join(self.dataset_type[self.gt_name][idx])
        sparse_depth = Image.open(sparse_depth)
        gt = Image.open(gt_name)
        w, h = sparse_depth.size
        sparse_depth, gt = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w), F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
        img = None
        if self.use_rgb:
            # assert len(self.dataset_type['img']) > 0, "No images in dataset dictionary"
            img = Image.open(self.dataset_type[self.img_name][idx])
            img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)
        # assert self.dataset_type['lidar_in'][idx].split('/')[-5] == self.dataset_type['img'][idx].split('/')[-4] == self.dataset_type['gt'][idx].split('/')[-5]
        # assert self.dataset_type['lidar_in'][idx].split('/')[-5] == self.dataset_type['img'][idx].split('/')[-4] == self.dataset_type['gt'][idx].split('/')[-5]
        # assert self.dataset_type['lidar_in'][idx].split('/')[-1].split('.')[0] == self.dataset_type['img'][idx].split('/')[-1].split('.')[0] == self.dataset_type['gt'][idx].split('/')[-1].split('.')[0]
        # assert self.dataset_type['lidar_in'][idx].split('/')[-2] == self.dataset_type['img'][idx].split('/')[-3] == self.dataset_type['gt'][idx].split('/')[-2]

        sparse_depth_np, gt_np, img_pil = self.define_transforms(sparse_depth, gt, img)
        sparse_depth.close()
        gt.close()
        input, gt = self.totensor(sparse_depth_np).float(), self.totensor(gt_np).float()

        if self.normal:
            # Put in {0-1} range and then normalize
            input = input/self.max_depth
            gt = gt/self.max_depth
            # input = self.depth_norm(input)

        if self.disp:
            input[input==0] = -1
            input = 1.0 / input
            input[input==-1] = 0
            gt[gt==0] = -1
            gt = 1.0 / gt
            gt[gt==-1] = 0

        if self.use_rgb:
            img_tensor = self.totensor(img_pil).float()
            if not self.normal:
                img_tensor = img_tensor*255.0
            # else:
                # img_tensor = self.normalize(img_tensor)
            input = torch.cat((input, img_tensor), 0)
        return input, gt
