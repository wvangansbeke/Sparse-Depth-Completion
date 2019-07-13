# Sparse-Depth-Completion
This repo contains the implementation of our paper [Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty](https://arxiv.org/abs/1902.05356) by [Wouter Van Gansbeke](https://github.com/wvangansbeke), Davy Neven, Bert De Brabandere and Luc Van Gool.

If you find this interesting or relevant to your work, consider citing:
```
@article{wvangansbeke_depth_2019,
  title={Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty},
  author={Van Gansbeke, Wouter and Neven, Davy and De Brabandere, Bert and Van Gool, Luc},
  journal={arXiv preprint arXiv:1902.05356},
  year={2019}
}
```

## Introduction
Monocular depth prediction methods fail to generate absolute and precise depth maps and stereoscopic approaches are still significantly outperformed by LiDAR based approaches. The goal of the depth completion task is to generate dense depth predictions from sparse and irregular point clouds. This project makes use of uncertainty to combine multiple sensor data in order to generate accurate depth predictions. Mapped lidar points together with RGB images (monococular) are used in this framework. This method holds the **1st place** entry on the [KITTI depth completion benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) at the time of submission of the paper.

The contribution of this paper is threefold:
* Global and local information are combined in order to accurately complete and correct the sparse and noisy LiDAR input. Monocular RGB images are used for the guidance of this depth completion task.
* Confidence maps are learned for the global branch and the local branch in an unsupervised manner. The predicted depth maps are weighted by their respective confidence map. This is the late fusion technique used in our framework.
* This method ranks first on the KITTI depth completion benchmark without using additional data or postprocessing.

See full demo on [YouTube](https://www.youtube.com/watch?v=Kr0W7io5rHw&feature=youtu.be). 

![demo](https://user-images.githubusercontent.com/9694230/51806092-db766c00-2275-11e9-8de0-888bed0fc9e8.gif)


## Requirements
Python 3.6 was used.
The most important packages are pytorch, torchvision, numpy, pillow and matplotlib.


## Dataset
The [Kitti dataset](www.cvlibs.net/datasets/kitti/) has been used. Once you've downloaded the dataset, you can find the required preprocessing in:
`Datasets/Kitti_loader.py`

Firstly, The png's are transformed to jpg - images to save place. Secondly, two directories are built i.e. one for training and one for validation.
The dataset consists of 85898 training samples, 6852 validation samples, 1000 selected validation samples and 1000 test samples.


## Run Code
To run the code:

`python main.py --data_path /path/to/data/ --lr_policy plateau`

Flags:
- Set flag "input_type" to rgb or depth.
- Set flag "pretrained" to true or false to use a model pretrained on Cityscapes for the global branch.
- See `python main.py --help` for more information.

## Trained models

You can find the model pretrained on Cityscapes [here](https://drive.google.com/drive/folders/1U7dvH4sC85KRVuV19fRpaMzJjE-m3D9x?usp=sharing). This model is used for the global network.

You can find a fully trained model for KITTI [here](https://drive.google.com/drive/folders/1U7dvH4sC85KRVuV19fRpaMzJjE-m3D9x?usp=sharing). The RMSE is around 802 mm on the selected validation set for this model as reported in the paper.

To test it: 
Save the model in a folder in the `Saved` directory.

and execute the following command:

`source Test/test.sh /path/to/dataset/ /path/to/directory_with_saved_model/ /path/to/directory_with_ground_truth_for_selected_validation_files/`

(You might have to recompile the C files for testing, provided by KITTI, if your architecture is different from mine)

## Results

Comparision with state-of-the-art:

![results](https://user-images.githubusercontent.com/9694230/59205060-49c32780-8ba2-11e9-8a87-34d8c3f99756.PNG)


## Discussion

Practical discussion:

- I recently increased the stability of the training process and I also made the convergence faster by adding some skip connections between the global and local network.
Initially I only used guidance by multiplication with an attention map (=probability), but found out that it is less robust and that differences between a focal MSE and vanilla MSE loss function were now negligible.
Be aware that this change will alter the appearance of the confidence maps since fusion happens at mutliple stages now.

- Feel free to experiment with different architectures for the global or local network. It is easy to add new architectures to `Models/__init__.py`

- I used a Tesla V100 GPU for evaluation.
