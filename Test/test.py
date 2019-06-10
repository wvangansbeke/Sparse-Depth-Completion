#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.transforms as transforms
import os, sys
from PIL import Image
import glob
import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
cwd = os.getcwd()
print(cwd)
import numpy as np
from Utils.utils import str2bool, AverageMeter, depth_read 
import Models
import Datasets
from PIL import ImageOps
import matplotlib.pyplot as plt
import time

#Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task TEST')
parser.add_argument('--dataset', type=str, default='kitti', choices = Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--mod', type=str, default='mod', choices = Models.allowed_models(), help='Model for use')
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')
parser.add_argument('--input_type', type=str, default='rgb', help='use rgb for rgbdepth')
# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')

# Paths settings
parser.add_argument('--save_path', type= str, default='../Saved/best', help='save path')
parser.add_argument('--data_path', type=str, required=True, help='path to desired datasets')
# parser.add_argument('--data_path', type=str, default='/esat/pyrite/wvangans/Datasets/KITTI/Depth_Completion/data/', help='path to desired datasets')

# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
parser.add_argument('--multi', type=str2bool, nargs='?', const=True, default=False, help="use multiple gpus")
parser.add_argument('--normal', type=str2bool, nargs='?', const=True, default=False, help="Normalize input")
parser.add_argument('--max_depth', type=float, default=85.0, help="maximum depth of input")
parser.add_argument('--sparse_val', type=float, default=0.0, help="encode sparse values with 0")


def main():
    global args
    global dataset
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn

    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]

    save_root = os.path.join(os.path.dirname(best_file_name), 'results')
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    print("==========\nArgs:{}\n==========".format(args))
    # INIT
    print("Init model: '{}'".format(args.mod))
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in)
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))
    if not args.no_cuda:
        # Load on gpu before passing params to optimizer
        if not args.multi:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(best_file_name):
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        model.load_state_dict(checkpoint['state_dict'])
        lowest_loss = checkpoint['loss']
        best_epoch = checkpoint['best epoch']
        print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))
        return

    if not args.no_cuda:
        model = model.cuda()
    print("Initializing dataset {}".format(args.dataset))
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)
    dataset.prepare_dataset()
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth_norm = transforms.Normalize(mean=[14.97/args.max_depth], std=[11.15/args.max_depth])
    model.eval()
    print("===> Start testing")
    total_time = []
    with torch.no_grad():
        for i, (img, rgb, gt) in tqdm.tqdm(enumerate(zip(dataset.selected_paths['lidar_in'],
                                           dataset.selected_paths['img'], dataset.selected_paths['gt']))):

            raw_path = os.path.join(img)
            raw_pil = Image.open(raw_path)
            gt_path = os.path.join(gt)
            gt_pil = Image.open(gt)
            assert raw_pil.size == (1216, 352)

            crop = 352-args.crop_h
            raw_pil_crop = raw_pil.crop((0, crop, 1216, 352))
            gt_pil_crop = gt_pil.crop((0, crop, 1216, 352))

            raw = depth_read(raw_pil_crop, args.sparse_val)
            raw = to_tensor(raw).float()
            gt = depth_read(gt_pil_crop, args.sparse_val)
            gt = to_tensor(gt).float()
            valid_mask = (raw > 0).detach().float()

            input = torch.unsqueeze(raw, 0).cuda()
            gt = torch.unsqueeze(gt, 0).cuda()

            if args.normal:
                # Put in {0-1} range and then normalize
                input = input/args.max_depth
                # input = depth_norm(input)

            if args.input_type == 'rgb':
                rgb_path = os.path.join(rgb)
                rgb_pil = Image.open(rgb_path)
                assert rgb_pil.size == (1216, 352)
                rgb_pil_crop = rgb_pil.crop((0, crop, 1216, 352))
                rgb = to_tensor(rgb_pil_crop).float()
                rgb = torch.unsqueeze(rgb, 0).cuda()
                if not args.normal:
                    rgb = rgb*255.0

                input = torch.cat((input, rgb), 1)

            torch.cuda.synchronize()
            a = time.perf_counter()
            output, _, _, _ = model(input)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total_time.append(b-a)
            if args.normal:
                output = output*args.max_depth
            output = torch.clamp(output, min=0, max=85)

            output = output * 256.
            raw = raw * 256.
            output = output[0][0:1].cpu()
            data = output[0].numpy()
    
            if crop != 0:
                padding = (0, 0, crop, 0)
                output = torch.nn.functional.pad(output, padding, "constant", 0)
                output[:, 0:crop] = output[:, crop].repeat(crop, 1)

            pil_img = to_pil(output.int())
            assert pil_img.size == (1216, 352)
            pil_img.save(os.path.join(save_root, os.path.basename(img)))
    print('average_time: ', sum(total_time[100:])/(len(total_time[100:])))
    print('num imgs: ', i + 1)


if __name__ == '__main__':
    main()
