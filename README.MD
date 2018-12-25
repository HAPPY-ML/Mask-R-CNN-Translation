# 实例分割 - 基于 Mask R-CNN 和 TensorFlow


[Temp Source](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)

今年11月，我们开源了我们对 [Mask R-CNN 实现源码](https://github.com/matterport/Mask_RCNN)。自那以后，我们的项目被fork了1400次，同时被应用在多个项目中，并且有许多的开发者对项目源码作出了改进。同时，我们也收到了许多关于我们项目的问题，所以在这篇文章中，我将阐述一下我们对 Mask R-CNN 的实现细节并展示它在实际项目中的应用。

本篇文章将覆盖以下两个内容：
1. Mask R-CNN 的概述。 
2. 如何从零开始训练一个数据模型并搭建一个图片颜色渲染滤镜。

> **代码提示:**
*这是本篇文章展示项目的[全部代码](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)，包括我创建的数据集和训练后的模型。*

## 什么是实例分割?
实例分割的任务就是，以像素为单位，识别图片中物体的轮廓。实力分割属于计算机视觉中最难实现的功能之一。以下是图像识别中和实例分割相关的几个任务，和他们处理后出来的结果：

![tasks](https://cdn-images-1.medium.com/max/1200/1*-zw_Mh1e-8YncnokbAFWxg.png)

- **图像分类 (Classification)**: 图片中有气球。
- **语义分割 (Semantic Segmentation)**: 这些是图片中组成所有气球的像素。
- **目标检测 (Object Detection)**: 这里是图片中7个气球的位置。我们需要识别被遮挡的物体。
- **实例分割 (Instance Segmentation)**: 这是是图片中7个气球的位置，包括组成每一个气球的像素。

## Mask R-CNN
Mask R-CNN (Regional Convolutional Neural Network: 区域卷积神经网络)，是一个分为两个主要步骤的框架：第一步，扫描整张图片，并生成可能包含所找实例的所有区域。第二步，将上一步生成的区域进行分类并生成边界框和蒙层

[Mask R-CNN](https://arxiv.org/abs/1703.06870) 是基于 [Faster R-CNN](https://arxiv.org/abs/1506.01497) 基础上的拓展，而且是由同一作者于去年发布。Faster R-CNN 是一个用于目标检测 (Object Detection) 的框架，而 Mask R-CNN 则是在这个基础上拓展出了实例分割 (Instance Segmentation) 与其他的新功能。

![Mask R-CNN](https://cdn-images-1.medium.com/max/1600/1*IWWOPIYLqqF9i_gXPmBk3g.png "Mask R-CNN framework. Source: https://arxiv.org/abs/1703.06870")

整体上，Mask R-CNN 可以分为以下几个模块：

## 1. Backbone

### Feature Pyramid Network

## 2. 区域建议网络 (Region Proposal Network 下面简称 RPN )
![rpn](https://cdn-images-1.medium.com/max/600/1*ESpJx0XLvyBa86TNo2BfLQ.png)

RPN 是一个通过使用滑动窗口方式扫描图片，以挖掘出含有物体区域的轻量级神经网络。

RPN 扫描出的区域,我们称为锚点。这些锚点在图上表现出来的就是一个个盒子(box)，如上图所示，即使这已经是简化过的图片。实际上，这些锚点能多达 200k 个，并且是不同尺寸，不同宽高比的。它们会重叠在一起，以尽可能多地覆盖图片。 

RPN 扫描出这么多的锚点，可以多快呢？事实上,相当快.这些滑动的窗口被RPN的卷积特性所处理, RPN 能在 GPU 上并行地扫描所有区域。更进一步，RPN并不是直接地扫描整张图片（虽然图例上我们是直接画出这些锚点）。相反， RPN 是通过 扫描骨干特征图（backbone feature map ） 得到锚点的。而这一的操作的目的是为了让 RPN 高效复用已经提取出的特征，以及避免重复的计算。根据 [Faster RCNN paper](https://arxiv.org/abs/1506.01497) 介绍， RPN 在这些优化下，能在 10 ms 内得到锚点。而在 Mask RCNN 中，我们通常会使用尺寸更大的图片以及更多的锚点，所以训练时间上可能会更久一点。

> **代码提示：**
> rpn 网络构建代码在 [`model.rpn_graph`](https://github.com/diaoxinqiang/Mask_RCNN/blob/e4b922624f0bd239607e7eeac79fa8dcab47b8b7/mrcnn/model.py) 方法中，可以在 [config.py](https://github.com/diaoxinqiang/Mask_RCNN/blob/e4b922624f0bd239607e7eeac79fa8dcab47b8b7/mrcnn/config.py) 文件中修改 `RPN_ANCHOR_SCALES` 以及   `RPN_ANCHOR_RATIOS` 参数改变锚点尺寸以及宽高比。

RPN 会为每个锚点生成两个输出:

1. 锚点类型（ Anchor Class ）: 前景类型 或 背景类型。前景类型就好比是一个在盒子里的物体。
2. 精细的边界框：一个前景锚点（也可以叫正极锚点）可能不会非常完美地定位在检测物体的中心位置上，所以 RPN 会估计一个估算出一个增量（ delta ，即针对矩形的x ,y , width , height 的百分比 ）来修正锚点盒子（ anchor box ）以更好地贴合检测物体。

![ Anchor Class](https://cdn-images-1.medium.com/max/600/1*EMNE8bxOT4RI3HMjIqjCwQ.png)
在 RPN 预测过程中，我们会选择最可能包含物体的锚点，然后去修正它们的位置以及尺寸。如果有些锚点重叠得太多，我们会保留前景得分最高的一个，而其他的锚点就丢弃掉（这种方式成为非最大抑制 Non-max Suppression ）。在我们得到最终的建议（也就是我们感兴趣的区域 regions of interest,简称 ROI）之后，我们就能进入下一阶段。

> **代码提示:**
`ProposalLayer` 是使用自定义的 Keras 层构建的， 用来读取 RPN的输出结果，选择得分最高的锚点以及进行边界框修正。
## 3. ROI Classifier & Bounding Box Regressor

### ROI Pooling

## 4. Segmentation Masks

## 一起来构造一个颜色渲染滤镜
和其他大多数有图片滤镜的图片编辑应用不同的是，我们的滤镜加了点人工智能，会更聪明一点：它会自动地找到特定的物体。如果你将这个滤镜在视频中使用，而不是单单一张图片，甚至会更有效果。
![效果图](https://cdn-images-1.medium.com/max/1200/1*lAP6vX1tLQaxFn6XGEQ32g.gif)

#### 1. Training Dataset
一般来说，我会从搜索包含我需要的物体的公开数据集开始。但在这个示例中，我想要把这个过程都记录夏日，以展示如何从零开始中构建一个数据集。
我从 `flickr` 中搜索关于气球的图片，并且这些图片是限定在"商业用户 & 允许修改"的许可证下。这个搜索结果得到的图片已经能满足我的需求了。我一共挑选了75张图片，并且它们分成训练集以及验证集。找图片是挺简单的事情，但标注图片才是艰难的部分。
![search](https://cdn-images-1.medium.com/max/1600/1*Q4tCdhwrklvJLM9zn5aDhg.png)
等等，难道我们不是应该需要上百万的图片去训练深度学习模型的吗？有时候，你确实需要如此，但是更多情况下，你并不需要这么多图片。我主要依靠下面两个技巧来显著地降低训练要求。
第一个技巧：迁移学习。简单来说，不是使用从零训练一个模型，而是已经在 `COCO dataset`中训练得到的模型参数中开始训练（我们在 `github` 代码库中有提供）。
Wait! Don’t we need, like, a million images to train a deep learning model? Sometimes you do, but often you don’t. I’m relying on two main points to reduce my training requirements significantly:

First, transfer learning. Which simply means that, instead of training a model from scratch, I start with a weights file that’s been trained on the COCO dataset (we provide that in the github repo). Although the COCO dataset does not contain a balloon class, it contains a lot of other images (~120K), so the trained weights have already learned a lot of the features common in natural images, which really helps. And, second, given the simple use case here, I’m not demanding high accuracy from this model, so the tiny dataset should suffice.

There are a lot of tools to annotate images. I ended up using VIA (VGG Image Annotator) because of its simplicity. It’s a single HTML file that you download and open in a browser. Annotating the first few images was very slow, but once I got used to the user interface, I was annotating at around an object a minute.
## Loading the Dataset



## Verify the Dataset

## Configurations

## Training

## Inspecting the Results

## Color Splash

## FAQ
