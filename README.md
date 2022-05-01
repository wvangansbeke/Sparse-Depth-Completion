# Sparse-Depth-Completion

This repo contains the implementation of our paper [Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty](https://arxiv.org/abs/1902.05356) by [Wouter Van Gansbeke](https://github.com/wvangansbeke), Davy Neven, Bert De Brabandere and Luc Van Gool.

If you find this interesting or relevant to your work, consider citing:

```
@inproceedings{wvangansbeke_depth_2019,
    author={Van Gansbeke, Wouter and Neven, Davy and De Brabandere, Bert and Van Gool, Luc},
    booktitle={2019 16th International Conference on Machine Vision Applications (MVA)},
    title={Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty},
    year={2019},
    pages={1-6},
    organization={IEEE}
}
```

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/)

## Introduction
Monocular depth prediction methods fail to generate absolute and precise depth maps and stereoscopic approaches are still significantly outperformed by LiDAR based approaches. The goal of the depth completion task is to generate dense depth predictions from sparse and irregular point clouds. This project makes use of uncertainty to combine multiple sensor data in order to generate accurate depth predictions. Mapped lidar points together with RGB images (monocular) are used in this framework. This method holds the **1st place** entry on the [KITTI depth completion benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) at the time of submission of the paper.

The contribution of this paper is threefold:
* Global and local information are combined in order to accurately complete and correct the sparse and noisy LiDAR input. Monocular RGB images are used for the guidance of this depth completion task.
* Confidence maps are learned for the global branch and the local branch in an unsupervised manner. The predicted depth maps are weighted by their respective confidence map. This is the late fusion technique used in our framework.
* This method ranks first on the KITTI depth completion benchmark without using additional data or postprocessing.

See full demo on [YouTube](https://www.youtube.com/watch?v=Kr0W7io5rHw&feature=youtu.be). The predictions of our model for the KITTI test set can be downloaded [here](https://drive.google.com/drive/folders/1U7dvH4sC85KRVuV19fRpaMzJjE-m3D9x).

![demo](https://user-images.githubusercontent.com/9694230/51806092-db766c00-2275-11e9-8de0-888bed0fc9e8.gif)


## Requirements
Python 3.7
The most important packages are pytorch, torchvision, numpy, pillow and matplotlib.
(Works with Pytorch 1.1)


## Dataset
The [Kitti dataset](www.cvlibs.net/datasets/kitti/) has been used. First download the dataset of the depth completion. Secondly, you'll need to unzip and download the camera images from kitti. 
I used the file `download_raw_files.sh`, but this is at your own risk. Make sure you understand it, otherwise don't use it. If you want to keep it safe, go to kitti's website. 

The complete dataset consists of 85898 training samples, 6852 validation samples, 1000 selected validation samples and 1000 test samples.

## Preprocessing
This step is optional, but allows you to transform the images to jpgs and to downsample the original lidar frames. This will create a new dataset in $dest.
You can find the required preprocessing in:
`Datasets/Kitti_loader.py`

Run:

`source Shell/preprocess $datapath $dest $num_samples`

(Firstly, I transformed the png's to jpg - images to save place. Secondly, two directories are built i.e. one for training and one for validation. See `Datasets/Kitti_loader.py`)

Dataset structure should look like this:
```
|--depth selection
|-- Depth
     |-- train
           |--date
               |--sequence1
               | ...
     |--validation
|--RGB
    |--train
         |--date
             |--sequence1
             | ...
    |--validation
```


## Run Code
To run the code:

`python main.py --data_path /path/to/data/ --lr_policy plateau`

Flags:
- Set flag "input_type" to rgb or depth.
- Set flag "pretrained" to true or false to use a model pretrained on Cityscapes for the global branch.
- See `python main.py --help` for more information.

or 

`source Shell/train.sh $datapath`

checkout more details in the bash file.

## Trained models
Our network architecture is based on [ERFNet](https://github.com/Eromera/erfnet_pytorch).

You can find the model pretrained on Cityscapes [here](https://drive.google.com/drive/folders/1U7dvH4sC85KRVuV19fRpaMzJjE-m3D9x?usp=sharing). This model is used for the global network.

You can find a fully trained model and its corresponding predictions for the KITTI test set [here](https://drive.google.com/drive/folders/1U7dvH4sC85KRVuV19fRpaMzJjE-m3D9x?usp=sharing). 
The RMSE is around 802 mm on the selected validation set for this model as reported in the paper. 

To test it: 
Save the model in a folder in the `Saved` directory.

and execute the following command:

`source Test/test.sh /path/to/directory_with_saved_model/ $num_samples /path/to/dataset/ /path/to/directory_with_ground_truth_for_selected_validation_files/`

(You might have to recompile the C files for testing, provided by KITTI, if your architecture is different from mine)

## Results

Comparision with state-of-the-art:

![results](https://user-images.githubusercontent.com/9694230/59205060-49c32780-8ba2-11e9-8a87-34d8c3f99756.PNG)


## Discussion

Practical discussion:

- I recently increased the stability of the training process and I also made the convergence faster by adding some skip connections between the global and local networks.
Initially I only used guidance by multiplication with an attention map (=probability), but found out that it is less robust and that differences between a focal MSE and vanilla MSE loss function were now negligible.
Be aware that this change will alter the appearance of the confidence maps since fusion happens at mutliple stages now.

- Feel free to experiment with different architectures for the global or local network. It is easy to add new architectures to `Models/__init__.py`

- I used a Tesla V100 GPU for evaluation.

## Acknowledgement
This work was supported by Toyota, and was carried out at the TRACE Lab at KU Leuven (Toyota Research on Automated Cars in Europe - Leuven)
