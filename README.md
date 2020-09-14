# Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images (ECCV 2020)

- [Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images (ECCV 2020)](#early-exit-or-not-resource-efficient-blind-quality-enhancement-for-compressed-images-eccv-2020)
  - [0. Background](#0-background)
  - [1. Pre-request](#1-pre-request)
    - [1.1. Environment](#11-environment)
    - [1.2. Data and pre-trained models](#12-data-and-pre-trained-models)
    - [1.3. Compress images](#13-compress-images)
  - [2. Test](#2-test)
  - [3. Training](#3-training)
  - [4. License & Citation](#4-license--citation)
  - [5. See more](#5-see-more)

## 0. Background

Official repository of [Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images](https://arxiv.org/abs/2006.16581), ECCV 2020.

- A **single blind** enhancement model for HEVC/JPEG-compressed images with a **wide range** of Quantization Parameters (QPs) or Quality Factors (QFs).
- A **multi-output dynamic** network with **early-exit** mechanism for easy input.
- A **Tchebichef-moments** based **NR-IQA** approach for early-exit decision. This IQA approach is highly interpretable and sensitive to blocking energy detection.

Feel free to contact: ryanxingql@gmail.com.

## 1. Pre-request

### 1.1. Environment

Python 3.6, PyTorch > 1.0, Pillow, Imageio:

```
conda create -n rbqe python=3.6 pillow=7.1.2 libtiff=4.1.0 imageio=2.9.0
conda activate rbqe
conda install -c pytorch pytorch=1.5
```

MATLAB R2019b.

### 1.2. Data and pre-trained models

All files below are prepared in [[Google Drive]](https://drive.google.com/drive/folders/16cAPczm_FQT5-U636QdzUXaQZikc6VeO?usp=sharing) (For Chinese researchers: [[百度网盘]](https://pan.baidu.com/s/1U9BtmZVxno_ZAON17XRBjg), 提取码rbqe). For demo, we prepare only 5 raw TIFF images.

### 1.3. Compress images

We use [RAISE](http://loki.disi.unitn.it/RAISE/) as raw image dataset. Download the TIFF images in RAISE, or prepare your own raw images.

**Overview**

- To generate HEVC-MSP-compressed test set:
  - center-crop raw images into `512x512`, considering that some compared approaches can not process larger images with prevalent GPUs.
  - stack images into a YUV video, which is convenient for compression by HM16.5.
  - compress this raw YUV video into 5 compressed YUV videos with 5 different QPs by HM16.5 (mode: main still picture, MSP). Therefore, each YUV video is a batch of images with the same QP.
- To generate JPEG-compressed test set:
  - compress each raw image into 5 compressed images with 5 different QFs by Python Pillow.
  - center-crop raw images into `512x512`.
  - stack images with the same QF into one YUV video. Therefore, 5 QFs correspond to 5 compressed YUV videos. Besides, raw images are also stacked into a raw YUV video (this video may be different from the raw video for HEVC experiment because the image order may be different).

**Specifically**

To generate HEVC-MSP-compressed test set:
1. `main_tiff2yuv420p.m`: Center-crop these images into `512x512` images, and stack them into a single YUV video.
2. `main_compress.bat` (Windows system): Compress this yuv with 5 different QPs: 22, 27, 32, 37 and 42. Then we get 5 YUV videos: `RAISE_qp22_512x512_test.yuv`, `RAISE_qp27_512x512_test.yuv`, `RAISE_qp32_512x512_test.yuv`, `RAISE_qp37_512x512_test.yuv`, and `RAISE_qp42_512x512_test.yuv`.

Note: you can also compress the YUV video on Ubuntu system using `main_compress.sh`.

To generate JPEG-compressed test set:
1. `python main_JPEG_compression.py`: Compress these images with 5 different QFs: 10, 20, 30, 40 and 50. Then we get 5 JPEG images for each raw image.
2. `main_jpeg2yuv420p.m`: Center-crop these images into `512x512` images, and stack them into 5 YUV videos: `RAISE_raw_512x512_test_jpeg.yuv`, `RAISE_qf10_512x512_test_jpeg.yuv`, `RAISE_qf20_512x512_test_jpeg.yuv`, `RAISE_qf30_512x512_test_jpeg.yuv`, `RAISE_qf40_512x512_test_jpeg.yuv`, and `RAISE_qf50_512x512_test_jpeg.yuv`.

## 2. Test

**Overview**

- Test all compressed YUV videos (actually compressed images in batches). For each compressed image, we obtain 5 enhanced images corresponding to 5 outputs of the network. Note that we do not use early-exit in this step, because we want to observe the PSNR vs. FLOPs performance under different threshold `T` in the next step. Therefore, the ave result in this step is not the final result.
- Evaluate quality score of each enhanced images by our Tchebichef-moments based IQA model.
- Generate the final PSNR vs. FLOPs result under one chosen threshold `T`.

**Specifically**

1. `python main_test.py -t HEVC -g 0`: test HEVC-compressed images, using gpu 0.
   Or: `python main_test.py -t JPEG -g 0`: test JPEG-compressed images, using gpu 0.
2. `main_cal_QualityScore.m`: change `type_test` into `HEVC` or `JPEG` in line 3, then run it by MATLAB.
3. `python main_tradeoff.py -t HEVC` or `python main_tradeoff.py -t JPEG`
   - EXP 1: Ablation. We force all QP=i (i=22,27,32,37,42) images to output at output=k (k=1,2,3,4,5), and observe the PSNR vs. FLOPs performance.
   - EXP 2: Tradeoff curve. We draw the PSNR vs. FLOPs under different threshold `T`.
   - EXP 3: Optimal result. As stated in our paper, we choose the turning point of the curve as the optimal `T`. Based on curves in EXP 2, we choose `T=0.84` for HEVC dataset and `T=0.67` for JPEG dataset.

Note: we use `T=0.89` for 1000-image HEVC dataset in our paper and `T=0.79` for 1000-image JPEG dataset in our paper. Here we have only 5 images, and the threshold `T` is re-chosen according to the curve.

Note 2: the curve may not be smooth, since we have only 5 images here.

## 3. Training

We are going to release a better version of training code in our extended work.

## 4. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

```tex
@InProceedings{RBQE,
author = {Xing, Qunliang and Xu, Mai and Li, Tianyi and Guan, Zhenyu},
title = {Early Exit Or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```

## 5. See more

- [PyTorch implementation of STDF (AAAI 2020)](https://github.com/RyanXingQL/STDF-PyTorch)
  - A **simple** yet **effective** video quality enhancement network.
  - Adopt **feature alignment** by multi-frame **deformable convolutions**, instead of motion estimation and motion compensation.

- [MFQEv2 (TPAMI 2019)](https://github.com/RyanXingQL/MFQEv2.0)
  - The first **multi-frame** quality enhancement approach for compressed videos.
  - The first to consider and utilize the **quality fluctuation** feature of compressed videos.
  - Enhance low-quality frames using **neighboring high-quality** frames.
