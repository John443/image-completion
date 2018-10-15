# image-completion

## Introduction

In this repo I tried to reimplement the paper [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/abs/1607.07539). Basically I trained a GAN and then trained a better z. I used z to fill into the pre-trained GAN model and generated the missing part of images.

## Network

Here I tried 2 types of GAN:
* DCGAN as the paper mentioned
* WGAN which is modified by myself

## How to use

* Preparation
	* Before we run the model, run `pip3 install -U face_recognition` to install a library to locate the human face of the image.

* Data
	* We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Randomly select 300 images from the dataset and run `python face_capture.py` to crop the human face part of the images and rescale it to 64 * 64.

* Train GAN
	* Run `python dcgan_test.py --mode gan` to train model by DCGAN, or `python wgan_test.py --mode gan` to train model by WGAN.

* Completion
	* Run `python dcgan_test.py --mode complete` to complete image by DCGAN model, or `python wgan_test.py --mode comlete` to complete by WGAN model.

## Analysis

In fact the code in this repo does not get the result I expected. There are following possible reasons:
1. The dataset is not good enough to train a GAN model. The missing part, epsecially the nose part, sometimes does not contain enough enformation for GAN to learn.
2. DCGAN itself is not stable enough to learn the facial information.
3. The face_recognition library doesn't get the exact facial part of an image.

## TODO

* Try to select high quality data to train a GAN model.
* Try to use object detection method to get facial information.
* Check if there is some implementation issue in GAN.
