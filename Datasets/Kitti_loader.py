"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re
import numpy as np
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.utils import write_file, depth_read
'''
attention:
    There is mistake in 2011_09_26_drive_0009_sync/proj_depth 4 files were
    left out 177-180 .png. Hence these files were also deleted in rgb
'''


class Random_Sampler():
    "Class to downsample input lidar points"

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, depth):
        mask_keep = depth > 0
        n_keep = np.count_nonzero(mask_keep)

        if n_keep == 0:
            return mask_keep
        else:
            depth_sampled = np.zeros(depth.shape)
            prob = float(self.num_samples) / n_keep
            mask_keep =  np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
            depth_sampled[mask_keep] = depth[mask_keep]
            return depth_sampled


class Kitti_preprocessing(object):
    def __init__(self, dataset_path, input_type='depth', side_selection=''):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.test_files = {'img': [], 'lidar_in': []}
        self.dataset_path = dataset_path
        self.side_selection = side_selection
        self.left_side_selection = 'image_02'
        self.right_side_selection = 'image_03'
        self.depth_keyword = 'proj_depth'
        self.rgb_keyword = 'Rgb'
        # self.use_rgb = input_type == 'rgb'
        self.use_rgb = True
        self.date_selection = '2011_09_26'

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                if re.search(self.depth_keyword, root):
                    self.train_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('velodyne_raw', root)
                                                        and re.search('train', root)
                                                        and re.search(self.side_selection, root)]))
                    self.val_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                              if re.search('velodyne_raw', root)
                                                              and re.search('val', root)
                                                              and re.search(self.side_selection, root)]))
                    self.train_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                          if re.search('groundtruth', root)
                                                          and re.search('train', root)
                                                          and re.search(self.side_selection, root)]))
                    self.val_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('groundtruth', root)
                                                        and re.search('val', root)
                                                        and re.search(self.side_selection, root)]))
                if self.use_rgb:
                    if re.search(self.rgb_keyword, root) and re.search(self.side_selection, root):
                        self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                               if re.search('train', root)]))
                                                               # and (re.search('image_02', root) or re.search('image_03', root))
                                                               # and re.search('data', root)]))
                       # if len(self.train_paths['img']) != 0:
                           # test = [os.path.join(root, file) for file in files if re.search('train', root)]
                        self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                            if re.search('val', root)]))
                                                            # and (re.search('image_02', root) or re.search('image_03', root))
                                                            # and re.search('data', root)]))
               # if len(self.train_paths['lidar_in']) != len(self.train_paths['img']):
                   # print(root)


    def downsample(self, lidar_data, destination, num_samples=500):
        # Define sampler
        sampler = Random_Sampler(num_samples)

        for i, lidar_set_path in tqdm.tqdm(enumerate(lidar_data)):
            # Read in lidar data
            name = os.path.splitext(os.path.basename(lidar_set_path))[0]
            sparse_depth = Image.open(lidar_set_path)


            # Convert to numpy array
            sparse_depth = np.array(sparse_depth, dtype=int)
            assert(np.max(sparse_depth) > 255)

            # Downsample per collumn
            sparse_depth = sampler.sample(sparse_depth)

            # Convert to img
            sparse_depth_img = Image.fromarray(sparse_depth.astype(np.uint32))

            # Save
            folder = os.path.join(*str.split(lidar_set_path, os.path.sep)[7:12])
            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)
            sparse_depth_img.save(os.path.join(destination, os.path.join(folder, name)) + '.png')

    def convert_png_to_rgb(self, rgb_images, destination):
        for i, img_set_path in tqdm.tqdm(enumerate(rgb_images)):
            name = os.path.splitext(os.path.basename(img_set_path))[0]
            im = Image.open(img_set_path)
            rgb_im = im.convert('RGB')
            folder = os.path.join(*str.split(img_set_path, os.path.sep)[8:12])
            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)
            rgb_im.save(os.path.join(destination, os.path.join(folder, name)) + '.jpg')
            # rgb_im.save(os.path.join(destination, name) + '.jpg')

    def get_selected_paths(self, selection):
        files = []
        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):
            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))
        return files

    def prepare_dataset(self):
        path_to_val_sel = 'depth_selection/val_selection_cropped'
        path_to_test = 'depth_selection/test_depth_completion_anonymous'
        self.get_paths()
        self.selected_paths['lidar_in'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'velodyne_raw'))
        self.selected_paths['gt'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'groundtruth_depth'))
        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
        self.test_files['lidar_in'] = self.get_selected_paths(os.path.join(path_to_test, 'velodyne_raw'))
        if self.use_rgb:
            self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
            self.test_files['img'] = self.get_selected_paths(os.path.join(path_to_test, 'image'))
            print(len(self.train_paths['lidar_in']))
            print(len(self.train_paths['img']))
            print(len(self.train_paths['gt']))
            print(len(self.val_paths['lidar_in']))
            print(len(self.val_paths['img']))
            print(len(self.val_paths['gt']))
            print(len(self.test_files['lidar_in']))
            print(len(self.test_files['img']))

    def compute_mean_std(self):
        nums = np.array([])
        means = np.array([])
        stds = np.array([])
        max_lst = np.array([])
        for i, raw_img_path in tqdm.tqdm(enumerate(self.train_paths['lidar_in'])):
            raw_img = Image.open(raw_img_path)
            raw_np = depth_read(raw_img)
            vec = raw_np[raw_np >= 0]
            # vec = vec/84.0
            means = np.append(means, np.mean(vec))
            stds = np.append(stds, np.std(vec))
            nums = np.append(nums, len(vec))
            max_lst = np.append(max_lst, np.max(vec))
        mean = np.dot(nums, means)/np.sum(nums)
        std = np.sqrt((np.dot(nums, stds**2) + np.dot(nums, (means-mean)**2))/np.sum(nums))
        return mean, std, max_lst


if __name__ == '__main__':

    # Imports
    import tqdm
    from PIL import Image
    import os
    import argparse
    from Utils.utils import str2bool

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument("--png2img", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--calc_params", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    parser.add_argument('--datapath', default='/usr/data/tmp/Depth_Completion/data')
    parser.add_argument('--dest', default='/usr/data/tmp/')
    args = parser.parse_args()

    dataset = Kitti_preprocessing(args.datapath, input_type='rgb')
    dataset.prepare_dataset()
    if args.png2img:
        os.makedirs(os.path.join(args.dest, 'Rgb'), exist_ok=True)
        destination_train = os.path.join(args.dest, 'Rgb/train')
        destination_valid = os.path.join(args.dest, 'Rgb/val')
        dataset.convert_png_to_rgb(dataset.train_paths['img'], destination_train)
        dataset.convert_png_to_rgb(dataset.val_paths['img'], destination_valid)
    if args.calc_params:
        import matplotlib.pyplot as plt
        params = dataset.compute_mean_std()
        mu_std = params[0:2]
        max_lst = params[-1]
        print('Means and std equals {} and {}'.format(*mu_std))
        plt.hist(max_lst, bins='auto')
        plt.title('Histogram for max depth')
        plt.show()
        # mean, std = 14.969576188369581, 11.149000139428104
        # Normalized
        # mean, std = 0.17820924033773314, 0.1327261921360489
    if args.num_samples != 0:
        print("Making downsampled dataset")
        os.makedirs(os.path.join(args.dest), exist_ok=True)
        destination_train = os.path.join(args.dest, 'train')
        destination_valid = os.path.join(args.dest, 'val')
        dataset.downsample(dataset.train_paths['lidar_in'], destination_train, args.num_samples)
        dataset.downsample(dataset.val_paths['lidar_in'], destination_valid, args.num_samples)
