# *Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images* (ECCV 2020)

## Background

Official repository of [*Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images*](https://arxiv.org/abs/2006.16581), ECCV 2020. [[速览]](https://github.com/ryanxingql/rbqe/blob/master/blog_zh.md)

- A **single blind** enhancement model for HEVC/JPEG-compressed images with a **wide range** of Quantization Parameters (QPs) or Quality Factors (QFs).
- A **multi-output dynamic** network with **early-exit** mechanism for easy input.
- A **Tchebichef-moments** based **NR-IQA** approach for early-exit decision. This IQA approach is highly interpretable and sensitive to blocking energy detection.

![network](https://user-images.githubusercontent.com/34084019/105739729-637dd200-5f73-11eb-923a-bb67ee9959eb.png)

Feel free to contact: `ryanxingql@gmail.com`.

## Code & Pre-trained Model

We have released two versions of the RBQE approach.

- [ECCV paper version](https://github.com/ryanxingql/rbqe/tree/34c961d4df7dea3882297601836b245d0b552739)
  - Adopts the RAISE data-set (high-quality and large-scale).
  - Adopts the HM software for compression and get YUV images.
  - Enhances only Y channel and report the Y-PSNR result (in accordance to previous papers).
  - Implements the image quality assessment module with MATLAB (convenient for visualization).
  - Assesses the Y quality; the IQA threshold is determined according to the Y performance.

- [Improved version](https://github.com/ryanxingql/powerqe)
  - Adopts the DIV2K data-set (used by most image restoration approaches).
  - Adopts the BPG software for compression (faster, but the result is different to that of the HM) and get PNG images (simpler).
  - Enhances RGB channels (more practical).
  - Re-implements the MATLAB-based image quality assessment module with Python (no need to use MATLAB anymore; but much, much slower than the MATLAB version).
  - Assesses the R quality; the IQA threshold is determined according to the R performance.

We have released the codes of all compared approaches in the latter repository.
