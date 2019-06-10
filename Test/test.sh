#!/bin/bash

echo 'Data path is: '$1
echo 'Save path is: '$2

python Test/test.py --data_path $1 --save_path Saved/$2 


# Arguments for evaluate_depth file: 
# - ground truth directory
# - results directory

Test/devkit/cpp/evaluate_depth ${3-/esat/pyrite/tmp/Depth_Completion/data/depth_selection/val_selection_cropped/groundtruth_depth} Saved/$2/results 
