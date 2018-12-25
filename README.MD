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

## 1. 骨干网络 (Backbone)
![Simplified illustration of the backbone nework](https://cdn-images-1.medium.com/max/1200/1*IDjLXsSw5QMFWDudayIBfw.png)

如图所示，是一个用常规的卷积神经网络实现的特征提取器，通常是采用的 ResNet50 或 ResNet101。神经网络前面的层级查找底层特征(图像的边缘与角)，接下来后面的层级会继续查找更高层级的特征(车、人、天空)。

通过骨干网络的处理，输入为 1024\*1024px\*3(RGB) 的图片会被转化成 32\*32\*2048 的特征点矩阵。输出的特征点矩阵就是我们后面步骤的输入。

> **代码提示:**
*骨干网络的构造在 [resnet_graph()](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py#L171) 方法中实现。代码同时支持 ResNet50 和 ResNet101.*

### 特征金字塔网络 (Feature Pyramid Network)
![Feature Pyrmaid Network](https://cdn-images-1.medium.com/max/1200/1*1sCveJrqfthOQsGGZRs2tQ.png)

由 Mask R-CNN 的作者提出的 [特征金字塔网络 (FPN)](https://arxiv.org/abs/1612.03144) 是基于上述骨干网络的优化。和骨干网络相比，FPN 可以在不同的层级更精准地表现出物体。

如图，在骨干网络的基础上，FPN 增加了另外一个金字塔结构，将高层级的输出特征传递到低层级。通过这种方式，每个层级都可以访问它的上一层或下一层特征。

我们 Mask RCNN 的实现采用的是 ResNet101 + FPN 作为骨干网络。

>**代码提示：**
*FPN 的构建是在 ResNet 的创建之后，在 [MaskRCNN.build()](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py#L1840) 方法中实现。
常规的骨干网络会输出单独一组特征矩阵，与此相比，RPN 会在新增金字塔的每一层分别产生一组特征矩阵。在使用中，我们会根据物体像素尺寸，动态地选择使用哪一组特征矩阵。在后面的文章中，我会直接描述成骨干网络的特征矩阵输出。但是请记住，使用 FPN 时，我们会得到多组特征矩阵输出，而我们会在其中挑选一组使用。*

## 2. 区域方案网络 (Region Proposal Network 下面简称 RPN )
![rpn](https://cdn-images-1.medium.com/max/600/1*ESpJx0XLvyBa86TNo2BfLQ.png)

RPN 是一个通过使用滑动窗口方式扫描图片，以挖掘出含有物体区域的轻量级神经网络。

RPN 扫描出的区域,我们称为锚点。这些锚点在图上表现出来的就是一个个盒子(box)，如上图所示，及时这已经是简化过的图片了实际上，这些锚点能多达 200k 个，并且是不同尺寸，不同宽高比的.它们会重叠在一起，以尽可能多地覆盖图片。 

RPN 扫描出这么多的锚点，可以多快呢？事实上,相当快.这些滑动的窗口被RPN的卷积特性所处理, RPN 能在 GPU 上并行地扫描所有区域。更进一步，RPN并不是直接地扫描整张图片（虽然图例上我们是直接画出这些锚点）。相反， RPN 是通过 扫描骨干特征图（backbone feature map ） 得到锚点的。而这一的操作的目的是为了让 RPN 高效复用已经提取出的特征，以及避免重复的计算。根据 [Faster RCNN paper](https://arxiv.org/abs/1506.01497) 介绍， RPN 在这些优化下，能在 10 ms 内得到锚点。而在 Mask RCNN 中，我们通常会使用尺寸更大的图片以及更多的锚点，所以训练时间上可能会更久一点。

> **代码提示：**
> rpn 网络构建代码在 [model.rpn_graph](https://github.com/diaoxinqiang/Mask_RCNN/blob/e4b922624f0bd239607e7eeac79fa8dcab47b8b7/mrcnn/model.py) 方法中，可以在 [config.py](https://github.com/diaoxinqiang/Mask_RCNN/blob/e4b922624f0bd239607e7eeac79fa8dcab47b8b7/mrcnn/config.py) 文件中修改 *RPN_ANCHOR_SCALES* 以及   *RPN_ANCHOR_RATIOS* 参数改变锚点尺寸以及宽高比。

RPN 会为每个锚点生成两个输出:

1. 锚点类型（ Anchor Class ）: 前景类型 或 背景类型。前景类型就好比是一个在盒子里的物体。
2. 精细的边界框：一个前景锚点（也可以叫正极锚点）可能不会非常完美地定位在检测物体的中心位置上，所以 RPN 会估计一个估算出一个增量（ delta ，即针对矩形的x ,y , width , height 的百分比 ）来修正锚点盒子（ anchor box ）以更好地贴合检测物体。

![ Anchor Class](https://cdn-images-1.medium.com/max/600/1*EMNE8bxOT4RI3HMjIqjCwQ.png)

## 3. ROI Classifier & Bounding Box Regressor

### ROI Pooling

## 4. Segmentation Masks
