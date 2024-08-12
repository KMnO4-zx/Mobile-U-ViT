# Mobile U-ViT: Revisiting large kernel \& U-shaped ViT for efficient medical image segmentation



Official pytorch code for "Mobile U-ViT: Revisiting large kernel \& U-shaped ViT for efficient medical image segmentation"



## Introduction
In practical clinical scenarios, the need for prompt execution of medical image analysis tasks on mobile devices with limited resources is evident. However, prevailing mobile models like MobileViT, originally designed for natural images, exhibit suboptimal performance on medical images. This deficiency arises from MobileViT's inherent limitations in capturing local intricacies and effectively integrating low-dimensional and high-dimensional information pertinent to medical imaging.  To rectify these shortcomings, we propose a mobile medical image segmentor termed Mobile U-shaped Vision Transformer (Mobile U-ViT). Specifically, the patch embedding part of Mobile U-ViT incorporates large convolution kernels and inverted pointwise bottlenecks to achieve larger receptive fields and improve feature extraction. Additionally, a cascaded decoder and downsampled skip-connections are proposed to facilitate the integration of low-dimensional and high-dimensional information. Furthermore, an adaptive Local-Global-Local block with a large kernel is introduced to facilitate efficient exchange of local-to-global information. Extensive experiments on five medical image datasets with three different modalities demonstrate the superiority of Mobile U-ViT over the state-of-the-art methods, while boasting lighter weights and a lower computational cost.

### Mobile U-ViT:

![framework](imgs/Mobile_U-ViT.png)

## Performance Comparison

<img src="imgs/better.png" title="preformance" style="zoom:12%;" align="left"/>

## Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 
```
└── MobileUtr
    ├── data
        ├── busi
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
    ├── dataloader
    ├── network
    ├── utils
    ├── main.py
    └── split.py
```
## Environment

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2
- albumentations: 1.2.0

## Training and Validation

You can first split your dataset:

```python
python split.py --dataset_name busi --dataset_root ./data
```

Then, train and validate:

```python
python main.py --model ["mobileuvit", "mobileuvit_l"] --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
```
