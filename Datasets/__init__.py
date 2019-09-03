"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from .Kitti_loader import *

dataset_dict = {'kitti': Kitti_preprocessing}

def allowed_datasets():
    return dataset_dict.keys()

def define_dataset(data, *args):
    if data not in allowed_datasets():
        raise KeyError("The requested dataset is not implemented")
    else:
        return dataset_dict['kitti'](*args)
    
