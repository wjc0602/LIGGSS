# Jittor 风格及语义引导的风景图片生成赛题
| 概览

![主要结果](pic\pipline.png)

| 主要结果

![主要结果](.selects\39589956_c7925b133a_b.jpg)![主要结果](.selects\389149475_87ddf6a45a_b.jpg)![主要结果](.selects\3602870053_a8e9eeb997_b.jpg)

## 简介

本项目包含了第三届计图挑战赛计图 - 风格及语义引导的风景图片生成赛题的代码实现。本项目的特点是：在Baseline的基础上添加了多阶段训练方法和分割模型，对baseline的掩码精度，风格相似度封指标有着很大的提升，是的生成的风景图像更加的美观。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型

模型名称：VGG19
使用数据集：ImageNet
代码链接：https://github.com/Jittor/jittor/blob/master/python/jittor/models/vgg.py
参数链接：jittorhub://vgg19.pkl

模型名称：vit_base_patch16_224_in21k
使用数据集：ImageNet
代码链接：在源代码中有使用jittor复现
参数链接：https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz


## 训练

单卡训练可运行以下命令：
```
python train.py
```
## 推理

生成测试集上的结果可以运行以下命令：

```
python test.py
```

## 致谢

此项目基于论文 *Semantic Image Synthesis with Spatially-Adaptive Normalization* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。