#!/bin/bash

echo 'Save path is: '$1
echo 'Data path is: '${3-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion/data/}

python Test/test.py --data_path ${3-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion/data/} --save_path Saved/$1 --num_samples ${2-0}

# Arguments for evaluate_depth file: 
# - ground truth directory
# - results directory

Test/devkit/cpp/evaluate_depth ${4-/esat/rat/wvangans/Datasets/KITTI/Depth_Completion/data/depth_selection/val_selection_cropped/groundtruth_depth} Saved/$1/results 
