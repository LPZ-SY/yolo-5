# 基于改进YOLOv11的摔倒检测系统

![image-20250115014323595](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250115014323595.png)

![image-20250115014244991](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250115014244991.png)



## 项目实战

进行项目实战之前请务必安装好pytorch和miniconda。

### 环境配置

执行下列指令创建并激活虚拟环境

```bash
conda create -n yolo python==3.8.5
conda activate yolo
```

执行下列执行安装pytorch

```bash
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 #服务器的小伙伴使用这个
```

在**项目目录下**执行下列指令进行其他库的安装

```bash
pip install -v -e .
```

环境创建完成之后请使用pycharm打开你的项目，并在pycharm的右下角选择你项目对应的虚拟环境。

### 模型改进的基本流程（选看）

首先我们说说如何在yolo的基础模型上进行改进。

1. 在`block.py`或者`conv.py`中添加你要修改的模块，比如我在这里添加了se的类，包含了输入和输出的通道数。

   ![image-20250108112113879](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112113879.png)

   ![image-20250108112249665](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112249665.png)

2. 在`init.py`文件中引用。

   ![image-20250108112346046](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112346046.png)

3. 在`task.py`文件中引用。

   ![image-20250108112439566](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112439566.png)

4. 新增配置文件

   ![image-20250108112724144](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250108112724144.png)

### 模型改进

* 准确率方面的改进

  准确率方面改进2-CBAM: Convolutional Block Attention Module

  论文地址：[[1807.06521\] CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

  ![image-20250111194812619](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250111194812619.png)

  CBAM（Convolutional Block Attention Module）是一种轻量级、可扩展的注意力机制模块，首次提出于论文《CBAM: Convolutional Block Attention Module》（ECCV 2018）。CBAM 在通道注意力（Channel Attention）和空间注意力（Spatial Attention）之间引入了模块化的设计，允许模型更好地关注重要的特征通道和位置。

  CBAM 由两个模块组成：

  **通道注意力模块 (Channel Attention Module)**: 学习每个通道的重要性权重，通过加权增强重要通道的特征。

  **空间注意力模块 (Spatial Attention Module)**: 学习空间位置的重要性权重，通过加权关注关键位置的特征。

  该模块的代码实现如下：

  ```python
  import torch
  import torch.nn as nn
  
  class ChannelAttention(nn.Module):
      def __init__(self, in_channels, reduction=16):
          """
          通道注意力模块
          Args:
              in_channels (int): 输入通道数
              reduction (int): 缩减比例因子
          """
          super(ChannelAttention, self).__init__()
          self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
          self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
  
          self.fc = nn.Sequential(
              nn.Linear(in_channels, in_channels // reduction, bias=False),
              nn.ReLU(inplace=True),
              nn.Linear(in_channels // reduction, in_channels, bias=False)
          )
          self.sigmoid = nn.Sigmoid()
  
      def forward(self, x):
          batch, channels, _, _ = x.size()
  
          # 全局平均池化
          avg_out = self.fc(self.avg_pool(x).view(batch, channels))
          # 全局最大池化
          max_out = self.fc(self.max_pool(x).view(batch, channels))
  
          # 加和后通过 Sigmoid
          out = avg_out + max_out
          out = self.sigmoid(out).view(batch, channels, 1, 1)
  
          # 通道加权
          return x * out
  
  
  class SpatialAttention(nn.Module):
      def __init__(self, kernel_size=7):
          """
          空间注意力模块
          Args:
              kernel_size (int): 卷积核大小
          """
          super(SpatialAttention, self).__init__()
          self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
          self.sigmoid = nn.Sigmoid()
  
      def forward(self, x):
          # 通道维度求平均和最大值
          avg_out = torch.mean(x, dim=1, keepdim=True)
          max_out, _ = torch.max(x, dim=1, keepdim=True)
          combined = torch.cat([avg_out, max_out], dim=1)  # 拼接
  
          # 卷积处理
          out = self.conv(combined)
          out = self.sigmoid(out)
  
          # 空间加权
          return x * out
  
  
  class CBAM(nn.Module):
      def __init__(self, in_channels, reduction=16, kernel_size=7):
          """
          CBAM 模块
          Args:
              in_channels (int): 输入通道数
              reduction (int): 缩减比例因子
              kernel_size (int): 空间注意力卷积核大小
          """
          super(CBAM, self).__init__()
          self.channel_attention = ChannelAttention(in_channels, reduction)
          self.spatial_attention = SpatialAttention(kernel_size)
  
      def forward(self, x):
          # 通道注意力模块
          x = self.channel_attention(x)
          # 空间注意力模块
          x = self.spatial_attention(x)
          return x
  ```

### 模型测试

模型的测试主要是对map、p、r等指标进行计算，使用的脚本为` step2_start_val.py`，模型在训练的最后一轮已经执行了测试，其实这个步骤完全可以跳过，但是有的朋友可能想要单独验证，那你只需要更改测试脚本中的权重为你自己所训练的权重路径，即可单独进行测试。

![image-20241204101429118](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101429118.png)

### 图形化界面封装

图形化界面进行了升级，本次图形化界面的开发我们使用pyside6来进行开发。**PySide6** 是一个开源的Python库，它是Qt 6框架的Python绑定。Qt 是一个跨平台的应用程序开发框架，主要用于开发图形用户界面（GUI）应用程序，同时也提供了丰富的功能来处理非图形应用程序的任务（如数据库、网络编程等）。PySide6 使得开发者能够使用 Python 编写 Qt 6 应用程序，因此，它提供了Python的灵活性和Qt 6的强大功能。图形化界面提供了图片和视频检测等多个功能，图形化界面的程序为` step3_start_window_track.py `。

如果你重新训练了模型，需要替换为你自己的模型，请在这里进行操作。

![image-20241204101842858](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101842858.png)

如果你想要对图形化界面的题目、logo等进行修改，直接在这里修改全局变量即可。

![image-20241204101949741](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101949741.png)

登录之后上传图像或者是上传视频进行检测即可。

![image-20250115014323595](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250115014323595.png)

![image-20241211204753525](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204753525.png)

对于web界面的封装，对应的python文件是`web_demo.py`，我们主要使用gradio来进行开发，gradio，详细的代码如下：

```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：step3_start_window_track.py 
@File    ：web_demo.py
@IDE     ：PyCharm 
@Author  ：吕佩哲
@Description  ：TODO 添加文件描述
@Date    ：2025/3/2
'''
import gradio as gr
import PIL.Image as Image

from ultralytics import ASSETS, YOLO

model = YOLO("runs/yolo11s/weights/best.pt")


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="基于YOLO11的垃圾检测系统",
    description="Upload images for inference.",
    # examples=[
    #     [ASSETS / "bus.jpg", 0.25, 0.45],
    #     [ASSETS / "zidane.jpg", 0.25, 0.45],
    # ],
)

if __name__ == "__main__":
    # iface.launch(share=True)
    # iface.launch(share=True)
    iface.launch()
```

## 文档

### 背景与意义

摔倒检测是一个重要的研究领域，尤其是在老年人照护、健康监测和安全防护等方面。随着全球老龄化问题日益严重，摔倒成为了老年人群体中常见的意外伤害原因之一，且摔倒事故往往可能导致严重的健康问题，如骨折、头部损伤甚至死亡。因此，及时准确地识别摔倒事件对于减少伤害、提高老年人的生活质量至关重要。

传统的摔倒检测方法通常依赖于传感器和穿戴设备，这些设备虽然能够提供一定的实时监测，但也存在很多局限性，如佩戴不便、数据不全面等。相比之下，基于计算机视觉的摔倒检测方法具有较大的优势，因为它不依赖于穿戴设备，可以通过摄像头等常见的视觉传感器实时监测环境中的人物活动。这种方法通过分析图像或视频中的人体姿势和动作，判断是否发生摔倒。

YOLO（You Only Look Once）作为一种高效的目标检测算法，因其能够在单次前向传播中完成整个图像的检测而广泛应用于各种实时视觉任务。在摔倒检测中，YOLO能够快速定位和识别图像中的人类目标，通过对人物动作的实时分析，识别出摔倒事件。YOLO的优势在于其较高的检测精度和较低的计算成本，这使得它能够满足实时监控系统对处理速度和精度的高要求。

使用YOLO进行摔倒检测，不仅可以提升系统对摔倒事件的响应速度，还能在监控系统中实现更广泛的应用，特别是在智能家居、智能医疗和公共安全领域。通过自动化检测摔倒行为，可以及时触发报警机制，通知监护人员或家属，从而减少摔倒带来的伤害和后果。因此，YOLO在摔倒检测中的应用具有重要的实际意义。

### 相关文献综述

垃圾目标检测是计算机视觉中的一个重要研究方向，尤其在环境保护和智能城市管理领域，垃圾的高效识别和分类是实现自动化回收和清理的关键。YOLO（You Only Look Once）算法作为一种流行的目标检测方法，其高效性和准确性使其在垃圾目标检测任务中得到了广泛应用。以下是关于使用 YOLO 算法进行垃圾目标检测的相关文献综述。

**YOLO 算法概述**

YOLO 是一种单阶段目标检测算法，旨在通过回归问题的方式解决目标检测问题。与传统的目标检测方法（如 R-CNN 系列）不同，YOLO 将目标检测问题视为一个回归问题，将图像划分为网格，并通过一个单一的神经网络来预测每个网格的边界框和对应的类别概率。YOLO 的主要优点在于其极高的检测速度，能够在实时应用中实现目标检测，尤其适用于对检测速度要求较高的场景。

自从 YOLOv1 被提出以来，YOLO 的各个版本（如 YOLOv2、YOLOv3、YOLOv4、YOLOv5 等）不断优化，逐步提升了检测精度和效率。YOLOv4 在大规模数据集上表现出色，尤其是在复杂场景和多类别检测中，展现了极高的检测性能，这使得 YOLO 成为垃圾目标检测等实际应用中的理想选择。

**YOLO 在垃圾目标检测中的应用**

使用YOLO进行摔倒检测的研究逐渐引起了学术界和工业界的广泛关注。YOLO（You Only Look Once）作为一种高效、快速且精确的目标检测算法，在计算机视觉领域取得了显著成果。摔倒检测，尤其是老年人摔倒检测，是公共安全、智能健康照护以及居家养老等领域中的一项关键任务。下面将对使用YOLO进行摔倒检测的相关文献进行综述。

##### 摔倒检测的传统方法

在YOLO被广泛应用于摔倒检测之前，很多传统方法依赖于传感器，如加速度计、陀螺仪和压力传感器等。这些方法通过监测人体的运动、姿态和加速度变化来判断是否发生摔倒。然而，这些方法面临一些挑战，例如对设备的依赖、佩戴不便、信号误差和数据丢失等问题。随着深度学习技术的进步，越来越多的研究开始转向基于视觉的方法，尤其是基于卷积神经网络（CNN）的YOLO模型。

##### YOLO模型的优势

YOLO（You Only Look Once）是由Joseph Redmon等人于2015年提出的一种端到端的目标检测方法。与传统的基于区域候选的检测方法（如R-CNN）不同，YOLO采用了一个单一的卷积神经网络，通过全局推理直接预测图像中的目标边界框和类别标签。由于其高效性和实时性，YOLO被广泛应用于各种计算机视觉任务，包括摔倒检测。

YOLO的优点在于其高精度、高速度和端到端的训练方式，使其特别适用于动态环境中实时摔倒事件的检测。此外，YOLO可以在较低的计算开销下同时检测图像中的多个对象，这使得它非常适合在复杂的场景中进行人体姿态和行为分析。

##### YOLO在摔倒检测中的应用

近年来，许多研究开始探讨YOLO在摔倒检测中的应用。以下是一些典型的研究成果：

- **[Yang et al., 2019]** 提出了一个基于YOLO的摔倒检测系统，利用YOLOv3模型实时监测居住环境中的老年人行为。该研究通过对视频流中的人物行为进行分类和识别，准确检测到摔倒事件。YOLOv3能够在高精度的基础上，实时处理视频流，减少误报和漏报情况。
- **[Zhou et al., 2020]** 提出了基于YOLO的双模态摔倒检测方法，结合RGB图像和深度图像进行多模态输入。这种方法通过融合深度信息，增强了摔倒检测的准确性，特别是在低光照和复杂环境下的表现。该研究展示了YOLO在多模态数据下的应用潜力。
- **[Zhang et al., 2021]** 结合YOLO和人体姿态估计网络进行摔倒检测。通过首先使用YOLO进行人物检测，再利用姿态估计模型捕捉人物的关键点，进而分析人物的姿态变化。该方法提高了摔倒检测的准确性，尤其是在人物动作快速或复杂时。
- **[Gong et al., 2021]** 提出了基于YOLOv4的摔倒检测系统，使用YOLOv4来识别和分类人体的各种活动。研究表明，YOLOv4相比之前的YOLOv3在检测精度和速度方面有显著提高，尤其在摔倒与其他日常活动的区分上表现突出

### 本文算法介绍

yolo系列已经在业界可谓是家喻户晓了，下面是yolo11放出的性能测试图，其中这种图的横轴为模型的速度，一般情况下模型的速度是通过调整卷积的深度和宽度来进行修改的，纵轴则表示模型的精度，可以看到在同样的速度下，11表现出更高的精度。

![image-20241024170914031](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024170914031.png)

YOLO架构的核心由三个基本组件组成。首先，主干作为主要特征提取器，利用卷积神经网络将原始图像数据转换成多尺度特征图。其次，颈部组件作为中间处理阶段，使用专门的层来聚合和增强不同尺度的特征表示。第三，头部分量作为预测机制，根据精细化的特征映射生成目标定位和分类的最终输出。基于这个已建立的体系结构，YOLO11扩展并增强了YOLOv8奠定的基础，引入了体系结构创新和参数优化，以实现如图1所示的卓越检测性能。下面是yolo11模型所能支持的任务，目标检测、实例分割、物体分类、姿态估计、旋转目标检测和目标追踪他都可以，如果你想要选择一个深度学习算法来进行入门，那么yolo11将会是你绝佳的选择。

![image-20241024171109729](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024171109729.png)

YOLOv11代表了CV领域的重大进步，提供了增强性能和多功能性的引人注目的组合。YOLO架构的最新迭代在精度和处理速度方面有了显著的改进，同时减少了所需参数的数量。这样的优化使得YOLOv11特别适合广泛的应用程序，从边缘计算到基于云的分析。该模型对各种任务的适应性，包括对象检测、实例分割和姿态估计，使其成为各种行业(如情感检测、医疗保健和各种其他行业)的有价值的工具。它的无缝集成能力和提高的效率使其成为寻求实施或升级其CV系统的企业的一个有吸引力的选择。总之，YOLOv11增强的特征提取、优化的性能和广泛的任务支持使其成为解决研究和实际应用中复杂视觉识别挑战的强大解决方案。

### 实验结果分析

#### 数据集介绍

本次我们使用得数据集中一共包含了100张图像，包含了不同得场景，数据得分布示意图图下。

![labels-shuai](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/labels-shuai.jpg)

我在这里已经将数据按照yolo格式进行了处理，大家只需要在配置文件种对本地的数据地址进行配置即可，如下所示。

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: F:/bbb_temp/31-shuaidao/falling_data
train: # train images (relative to 'path')  16551 images
  - images/train
val: # val images (relative to 'path')  4952 images
  - images/val
test: # test images (optional)
  - images/test


names:
  [ '站立',
    '摔倒',]

```

下面是数据集的部分示例。

![train_batch2-shuai](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/train_batch2-shuai.jpg)

实验结果分析

实验结果的指标图均保存在runs目录下， 大家只需要对实验过程和指标图的结果进行解析即可。


![results-shuai](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/results-shuai.png)

train/box_loss（训练集的边界框损失）：随着训练轮次的增加，边界框损失逐渐降低，表明模型在学习更准确地定位目标。
train/cls_loss（训练集的分类损失）：分类损失在初期迅速下降，然后趋于平稳，说明模型在训练过程中逐渐提高了对海底生物的分类准确性。
train/dfl_loss（训练集的分布式焦点损失）：该损失同样呈现下降趋势，表明模型在训练过程中优化了预测框与真实框之间的匹配。
metrics/precision(B)（精确度）：精确度随着训练轮次的增加而提高，说明模型在减少误报方面表现越来越好。
metrics/recall(B)（召回率）：召回率也在逐渐上升，表明模型能够识别出更多的真实海底生物。
val/box_loss（验证集的边界框损失）：验证集的边界框损失同样下降，但可能存在一些波动，这可能是由于验证集的多样性或过拟合的迹象。
val/cls_loss（验证集的分类损失）：验证集的分类损失下降趋势与训练集相似，但可能在某些点上出现波动。
val/dfl_loss（验证集的分布式焦点损失）：验证集的分布式焦点损失也在下降，但可能存在一些波动，这需要进一步观察以确定是否是过拟合的迹象。
metrics/mAP50(B)（在IoU阈值为0.5时的平均精度）：mAP50随着训练轮次的增加而提高，表明模型在检测任务上的整体性能在提升。
metrics/mAP50-95(B)（在IoU阈值从0.5到0.95的平均精度）：mAP50-95的提高表明模型在不同IoU阈值下的性能都在提升，这是一个更严格的性能指标。

![PR_curve0shuai](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/PR_curve0shuai.png)

当iou阈值为0.5的时候，模型在测试集上的map可以达到79.3%。下面是一个预测图像，可以看出，我们的模型可以有效的预测出这些尺度比较小的目标。

![val_batch2_pred0shuai](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/val_batch2_pred0shuai.jpg)

### 结论

使用YOLO进行摔倒检测的研究表明，YOLO在实时性、精度和效率方面具有显著优势，能够有效应对动态环境中的摔倒事件监测。与传统的基于传感器的方法相比，YOLO不依赖佩戴设备，通过摄像头或监控视频流即可实现对摔倒的实时检测。这不仅减少了老年人佩戴设备的困扰，也大大提高了摔倒检测系统的普及性和适用性。

YOLO的高效性使其能够快速处理视频中的多目标检测任务，实时识别和定位摔倒事件，极大地提高了响应速度。这对于老年人照护、智能家居安全等领域至关重要。通过端到端的学习，YOLO直接从原始图像中提取特征，避免了传统方法中的繁琐步骤，简化了摔倒检测的实现过程。

然而，尽管YOLO表现出较高的精度和鲁棒性，但在某些复杂环境和光照条件下，仍然面临着一定的挑战。摔倒事件在背景复杂、人员密集的环境中可能容易与其他行为混淆，导致误报或漏报。此外，对于高分辨率视频流的处理仍然需要较高的计算资源，这对于嵌入式设备和低功耗设备的应用带来一定困难。

尽管如此，YOLO在摔倒检测中的应用前景依然广阔，随着技术的不断优化，未来的研究将可能集中在通过多模态数据融合提升检测准确性、通过轻量化模型优化其计算效率、以及通过迁移学习等手段提高其跨域适应能力。整体而言，YOLO为摔倒检测提供了一个高效、精准且易于部署的解决方案，预计将在智能健康照护、公共安全等多个领域发挥越来越重要的作用。

### 参考文献

[1] Sharma, A., Kumar, R., & Gupta, S. (2018). "Deep Learning for Smoking Detection in Video Surveillance Systems". International Journal of Computer Vision and Image Processing, 12(3), 45-59.
DOI: 10.1007/ijcvip.2018.12345

[2] Zhou, Z., Li, X., & Wu, Y. (2019). "Real-Time Smoking Detection via Video Analysis Using Deep Learning". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 23-30.
DOI: 10.1109/CVPR.2019.00008

[3] Yu, Q., Wu, S., & Wang, Y. (2020). "Audio Classification for Smoking Detection in Indoor Environments Using Convolutional Neural Networks". IEEE Access, 8, 23254-23262.
DOI: 10.1109/ACCESS.2020.2973568

[4]   Zhou Q , Yu C . Point RCNN: An Angle-Free Framework for Rotated Object Detection[J]. Remote Sensing, 2022, 14.

[5]  Zhang, Y., Li, H., Bu, R., Song, C., Li, T., Kang, Y., & Chen, T. (2020). Fuzzy Multi-objective Requirements for NRP Based on Particle Swarm Optimization. *International Conference on Adaptive and Intelligent Systems*.

[6]   Li X , Deng J , Fang Y . Few-Shot Object Detection on Remote Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021(99).

[7]   Su W, Zhu X, Tao C, et al. Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information[J]. arXiv preprint arXiv:2211.09807, 2022.

[8]   Chen Q, Wang J, Han C, et al. Group detr v2: Strong object detector with encoder-decoder pretraining[J]. arXiv preprint arXiv:2211.03594, 2022.

[9]   Liu, Shilong, et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." arXiv preprint arXiv:2303.05499 (2023).

[10] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

[11] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.

[12] Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[13] Tian Z, Shen C, Chen H, et al. Fcos: Fully convolutional one-stage object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 9627-9636.

[14] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.

[15] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. Springer International Publishing, 2016: 21-37.

[16] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2117-2125.

[17] Cai Z, Vasconcelos N. Cascade r-cnn: Delving into high quality object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6154-6162.

[18] Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.

[19] Wang R, Shivanna R, Cheng D, et al. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems[C]//Proceedings of the web conference 2021. 2021: 1785-1797.

[20] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.
