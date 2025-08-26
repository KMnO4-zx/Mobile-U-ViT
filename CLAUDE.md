# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库工作时提供指导。

## 项目概述
Mobile U-ViT 是一个专为移动设备医学影像分割设计的轻量级 Vision Transformer 架构。通过创新的 ConvUtr 块和 LGL（大核局部-全局-局部）模块，结合了 CNN 的效率和 Transformer 的能力。

## 架构特点
- **U形编码器-解码器** 结构，带跳跃连接
- **ConvUtr 块**: 参数高效的 CNN 块，使用大核卷积
- **LGL 模块**: 混合局部-全局-局部注意力机制
- **移动优先设计**: 为资源受限设备优化

## 核心目录
- `network/MobileUViT.py`: 核心模型架构
- `dataloader/dataset.py`: 医学数据集处理
- `utils/`: 损失函数和评估指标
- `main.py`: 训练和验证流程

## 常用命令
```bash
# 数据集准备
python split.py --dataset_name busi --dataset_root ./data

# 训练基础模型
python main.py --model mobileuvit --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt

# 训练大模型
python main.py --model mobileuvit_l --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
```

## 模型变体
- `mobileuvit`: 基础模型，通道维度[16,32,64,128]
- `mobileuvit_l`: 大模型，通道维度[32,64,128,256]

## 输入输出格式
- **输入**: RGB图像 [B,3,256,256] 或灰度图转3通道
- **输出**: 分割掩码 [B,1,512,512]
- **数据集结构**: `data/{数据集名}/images/` 和 `data/{数据集名}/masks/0/`

## 依赖环境
- PyTorch 1.13.0 + CUDA 11.7
- albumentations 1.2.0
- scikit-learn 1.0.2
- timm (用于 DropPath 和层操作)

## 训练配置
- **优化器**: SGD，动量0.9
- **损失函数**: BCEDiceLoss
- **学习率调度**: 多项式衰减 (幂次=0.9)
- **训练轮数**: 300轮
- **批次大小**: 8 (根据GPU内存调整)
- 请确保使用 uv 来管理任何 python 的依赖