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
+ utils.py: 项目实现过程中用到的一些工具，如对音频数据预处理切割为相同长度的音频，绘制loss图等

## 数据集

语音唤醒任务：

+ MobvoiHotwords: 是从Mobvoi的商业智能扬声器收集的唤醒单词的语料库
+ 由“ Hi xiaowen”和“ Nihao Wenwen”的关键字语音，以及非关键字语音组成
+ [国内镜像](https://link.ailemon.me/?target=http://openslr.magicdatatech.com/resources/87/mobvoi_hotword_dataset.tgz)

关键词识别任务：

+ 包含bed, bird, cat, dog, eight, five, stop, wow, zero等三十种关键词语音数据

+ [下载地址]([Synthetic Speech Commands Dataset | Kaggle](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset))

