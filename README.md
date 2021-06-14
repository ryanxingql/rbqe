# *Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images* (ECCV 2020)

## 0. Background

Official repository of [*Early Exit or Not: Resource-Efficient Blind Quality Enhancement for Compressed Images*](https://arxiv.org/abs/2006.16581), ECCV 2020. [[速览 (中文)]](https://github.com/RyanXingQL/Blog/blob/main/posts/rbqe.md)

- A **single blind** enhancement model for HEVC/JPEG-compressed images with a **wide range** of Quantization Parameters (QPs) or Quality Factors (QFs).
- A **multi-output dynamic** network with **early-exit** mechanism for easy input.
- A **Tchebichef-moments** based **NR-IQA** approach for early-exit decision. This IQA approach is highly interpretable and sensitive to blocking energy detection.

![network](https://user-images.githubusercontent.com/34084019/105739729-637dd200-5f73-11eb-923a-bb67ee9959eb.png)

Feel free to contact: `ryanxingql@gmail.com`.

## 1. Codes & Pre-trained Models

[[Previous Version]](https://github.com/RyanXingQL/RBQE/tree/34c961d4df7dea3882297601836b245d0b552739)

To unify most of the quality enhancement approaches, we have released the improved RBQE at [PowerQE](https://github.com/RyanXingQL/PowerQE). Codes of all compared approaches are also presented there.

A Python-based image quality assessment module (IQAM) is provided at [PowerQE](https://github.com/RyanXingQL/PowerQE). The MATLAB-based IQAM is provided at this repository, which is much, much faster.

## 2. Difference from the Paper

1. **Dataset**. In the paper, we use the high-resolution RAISE dataset. In [PowerQE](https://github.com/RyanXingQL/PowerQE), the commonly-used DIV2K dataset is adopted for all approaches.
2. **Image compression**. In the paper, we use HM software to obtain HEVC-compressed images. In [PowerQE](https://github.com/RyanXingQL/PowerQE), BPG is adopted to obtain compressed images, which is simpler.
3. **YCbCr or RGB**. In the paper, we only enhance the Y channel and report the Y-PSNR result. In [PowerQE](https://github.com/RyanXingQL/PowerQE), since the input images are with PNG format, we enhance all R, G and B channels.
4. **Image quality assessment (IQA)**. In the paper, the IQA is conducted on the Y channel, and the threshold of IQA module (IQAM) is determined according to the Y performance. In [PowerQE](https://github.com/RyanXingQL/PowerQE), we conduct IQA on R channel for simplicity and then set the threshold.
5. **IQA implementation**. In [PowerQE](https://github.com/RyanXingQL/PowerQE), a Python-based IQAM is provided for an end-to-end Python experience. In the paper, we use a MATLAB-based IQAM, which is much faster but independent of the Python-based enhancement model. The MATLAB-based IQAM is provided at this repository. Note that the thresholds are different between two versions.

## 3. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

```tex
@incollection{RBQE_xing_2020,
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
