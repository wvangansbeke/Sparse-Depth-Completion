# Sparse-Depth-Completion
This repo contains the implementation of our paper Sparse and noisy LiDAR Completion with RGB Guidance and Uncertainty by [Wouter Van Gansbeke](https://github.com/wvangansbeke), Davy Neven, Bert De Brabandere and Luc Van Gool.

## Introduction
Monocular depth prediction methods fail to generate absolute and precise depth maps and stereoscopic approaches are still significantly outperformed by LiDAR based approaches. The goal of the depth completion task is to generate dense depth predictions from sparse and irregular point clouds. This project makes use of uncertainty to combine multiple sensor data in order to generate accurate depth predictions. Mapped lidar points together with RGB images (monococular) are used in this framework. This method holds the **1st place** entry on the KITTI depth completion benchmark. Code will be released soon.

See full demo on [YouTube](https://www.youtube.com/watch?v=Kr0W7io5rHw&feature=youtu.be). 

![demo](https://user-images.githubusercontent.com/9694230/51806092-db766c00-2275-11e9-8de0-888bed0fc9e8.gif)
