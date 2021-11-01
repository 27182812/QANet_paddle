# QANET_paddle

[English](./README.md) | 简体中文

## 一、简介

本项目基于paddlepaddle_v2.2.0rc0框架复现QANET(ICLR 2018)[链接 ](https://arxiv.org/abs/1804.09541)。

目前的端到端机器阅读和问答模型主要基于包含注意力的循环神经网络，抛开优点，这些模型的主要缺点是：在训练和推理方面效率较低。 因此作者提出了一种名为QANet的问答架构，这个网络不需要使用递归网络，它的编码器完全由卷积和self-attention组成，卷积网络处理局部信息，self-attention处理全局范围内的信息。

**论文:**

- [1] Yu, A. W. ,  D  Dohan,  Luong, M. T. ,  Zhao, R. ,  Chen, K. , &  Norouzi, M. , et al. (2018). Qanet: combining local convolution with global self-attention for reading comprehension.

**参考项目：**

- [BangLiu/QANet-PyTorch: Re-implement "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension" (github.com)](https://github.com/BangLiu/QANet-PyTorch)

**项目aistudio地址：**

- notebook任务：

## 二、复现精度+对齐

该模型是在SQuAD1.1数据集上训练＋验证。

！！！指标and结果

对齐具体步骤可参考https://github.com/PaddlePaddle/models/blob/develop/docs/ThesisReproduction_CV.md

### step1: 模型结构对齐 

对齐模型结构时，一般有3个主要步骤：

- 网络结构代码转换
- 权重转换
- 模型组网正确性验证

对齐结果如下所示：

![对齐1图](imgs/1.png)

### step2: 验证/测试集数据读取对齐

对齐结果如下所示：

![对齐2图](imgs/2.png)

### step3: 评估指标对齐

对齐结果如下所示：

![对齐3图](imgs/3.png)

### step4: 损失函数对齐

### ![对齐4图](imgs/4.png)

### step5: 反向对齐

![对齐5图](imgs/5.png)



## 三、数据集

[SQuAD1.1](https://datarepository.wolframcloud.com/resources/SQuAD-v1.1)

- 数据集大小：
  - 训练集：87.5K
  - 验证集：10.1K
- 数据格式：文本，JSON

## 四、环境依赖

- 硬件：CPU、GPU（建议使用16G及以上）
- 框架：
  - PaddlePaddle >= 2.2.0rc0
- 包：
  - spacy
  - ujson
