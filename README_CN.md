# KWS_pytorch
Keyword spotting, Speech wake_up, pytorch, DNN, CNN, TDNN, DFSMN, LSTM 


## 项目介绍

+ 该项目基于论文[**Small-footprint keyword spotting using deep neural networks**](https://ieeexplore.ieee.org/abstract/document/6854370/)

+ 可使用多种神经网络实现，如DNN, CNN, TDNN, DFSMN, LSTM等

+ 可以用来做关键词识别或语音唤醒任务

## 项目文件

+ dataset.py: 定义类KWSDataset提取数据集，其中数据集格式为：path/关键词/语音文件
+ model.py: 定义了各种神经网络的具体结构，如DNN, CNN, TDNN, DFSMN, LSTM等
+ train.py: 训练数据的方法train，和测试数据的方法test
+ run.py: 主程序，通过其运行整个识别任务，在其中可设置训练所需的各种参数
+ utils: 项目实现过程中用到的一些工具，如对音频数据预处理切割为相同长度的音频，绘制loss图等

## 数据集

语音唤醒任务：

+ MobvoiHotwords: 是从Mobvoi的商业智能扬声器收集的唤醒单词的语料库
+ 由“ Hi xiaowen”和“ Nihao Wenwen”的关键字语音，以及非关键字语音组成
+ [国内镜像](https://link.ailemon.me/?target=http://openslr.magicdatatech.com/resources/87/mobvoi_hotword_dataset.tgz)

关键词识别任务：

+ 包含bed, bird, cat, dog, eight, five, stop, wow, zero等三十种关键词语音数据

+ [下载地址]([Synthetic Speech Commands Dataset | Kaggle](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset))


## 实验结果

关键词识别任务：
+ STFT：

![acc](https://user-images.githubusercontent.com/63407850/158146828-052632ab-4b8c-4e25-acac-337d4ca51896.png)
+ Deep KWS:

![acc_KWS](https://user-images.githubusercontent.com/63407850/158160063-43cf819b-f47d-41df-bbf6-bf1038c901e2.png)

+ Table of Accuary: 

| Module   | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 | text   |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| DNN      | 38.57% | 52.85% | 58.81% | 67.48% | 71.00% | 62.59% |
| CNN      | 95.30% | 96.12% | 96.30% | 97.20% | 96.75% | 95.17% |
| TDNN     | 70.10% | 69.02% | 74.35% | 77.87% | 80.76% | 76.50% |
| LSTM     | 57.36% | 74.35% | 75.16% | 79.31% | 81.39% | 78.75% |
| DFSMN    | 91.15% | 92.14% | 94.94% | 93.86% | 94.04% | 90.34% |
| DNN(KWS) | 87.97% | 90.37% | 91.04% | 91.18% | 90.44% | 89.67% |

**语音唤醒：**

+ 基于Deep_KWS的Accuracy:

![acc](https://user-images.githubusercontent.com/63407850/158172843-73ad9507-d7fe-4087-898a-4afbec1d3278.png)

+ 基于Deep_KWS的Loss:

![loss](https://user-images.githubusercontent.com/63407850/158172839-e1a75c50-bfed-485c-9e69-d552d96e05f0.png)
