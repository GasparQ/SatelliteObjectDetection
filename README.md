# Satellite Object Detection

## Introduction

This repository will register some techniques to perform object detection over satellite images.

Several experiments will be proposed from pytorch neural networks.

The goal here is not to provide a reusable package but either a knowledge library to regroup multiple techniques selected over the internet.

## Setup

Before running any code, you will have to create a conda environment from the file `environment.yml`:
```bash
conda env create -f environment.yml
```

## Datasets

Through this repository we will test many different challenges and used the following datasets:
- [DOTA](https://captain-whu.github.io/DOTA/dataset.html) (1.0): Satellite images from multiple satellites
- [iSAID](https://captain-whu.github.io/iSAID/) (no version provided - 2022/05): Semantic masks of the DOTA images

## UNet

The folder `sod/unet` contains the code to train & test a UNet with pytorch from this [source tutorial](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/).