# *Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images* (ECCV 2020)

## 0. Background

Official repository of [*Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images*](https://arxiv.org/abs/2006.16581), ECCV 2020. [[速览]](https://github.com/ryanxingql/blog/blob/main/posts/rbqe.md)

- A **single blind** enhancement model for HEVC/JPEG-compressed images with a **wide range** of Quantization Parameters (QPs) or Quality Factors (QFs).
- A **multi-output dynamic** network with **early-exit** mechanism for easy input.
- A **Tchebichef-moments** based **NR-IQA** approach for early-exit decision. This IQA approach is highly interpretable and sensitive to blocking energy detection.

![network](https://user-images.githubusercontent.com/34084019/105739729-637dd200-5f73-11eb-923a-bb67ee9959eb.png)

Feel free to contact: `ryanxingql@gmail.com`.

## 1. Code & Pre-trained Model

We have released two versions of the RBQE approach.

1. [[ECCV paper version]](https://github.com/ryanxingql/rbqe/tree/34c961d4df7dea3882297601836b245d0b552739)
   1. adopts the RAISE data-set (high-quality and large-scale).
   2. adopts the HM software for compression and get YUV images.
   3. enhances only Y channel and report the Y-PSNR result (in accordance to previous papers).
   4. implements the image quality assessment module with MATLAB (convenient for visualization).
   5. assesses the Y quality; the IQA threshold is determined according to the Y performance.

2. [[Improved version]](https://github.com/ryanxingql/powerqe)
   1. adopts the DIV2K data-set (used by most image restoration approaches).
   2. adopts the BPG software for compression (faster, but the result is different to that of the HM) and get PNG images (simpler).
   3. enhances RGB channels (more practical).
   4. re-implements the MATLAB-based image quality assessment module with Python (much, much slower than the MATLAB version).
   5. assesses the R quality; the IQA threshold is determined according to the R performance.

We have released the codes of all compared approaches in the latter repository.

## 2. License

We adopt Apache License v2.0.

If you find this repository helpful, you may cite:

```tex
@incollection{2020xing,
	doi = {10.1007/978-3-030-58517-4_17},
	url = {https://doi.org/10.1007%2F978-3-030-58517-4_17},
	year = 2020,
	publisher = {Springer International Publishing},
	pages = {275--292},
	author = {Qunliang Xing and Mai Xu and Tianyi Li and Zhenyu Guan},
	title = {Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images},
	booktitle = {Computer Vision {\textendash} {ECCV} 2020}
}
```
