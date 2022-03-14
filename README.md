# KWS_pytorch

Keyword spotting, Speech wake_up, pytorch, DNN, CNN, TDNN, DFSMN, LSTM 


## Introduction

+ The project is based on ICASSP 2014 paper [**Small-footprint keyword spotting using deep neural networks**](https://ieeexplore.ieee.org/abstract/document/6854370/).

+ We implement the idea with various deep neural network architecture, e.g.,DNN, CNN, TDNN, DFSMN, LSTM.

+ The project can be applied to several tasks, such as key-word spotting and speech wake-up.

## Documents

+ command_loader.py: CommandLoader is defined for data extraction. The data is structured as follow
  + path/key words/audio file (.wav)

+ model.py: Implementation of several backbones: DNN, CNN, TDNN, DFSMN, LSTM.
+ train.py: Definition of training & testing process.
+ run.py: Main program for training & testing. Possible parameters are explained below:
  + 


## Datasets

**Speech wake-up：**

+ MobvoiHotwords: A corpus of wake-up words collected from a commercial smart speaker of Mobvoi.
  + Containing audio of "Hi xiaowen" and " Nihao Wenwen", as well as noise speech.
  + [Homepage](https://www.openslr.org/87)

**Key-word spotting：**

+ Synthetic Speech Commands Dataset. 
  + Consisted of key-words audio of thirty categories, e.g., "bed", "bird", "cat", "dog", "eight", "five", "stop", "wow", "zero".
  + [Download link](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset)


## Visualization of Results

**Key-word spotting：**

+ Batchsize-Accuracy curve with STFT：

![acc](https://user-images.githubusercontent.com/63407850/158172947-82f73d51-4949-414c-8110-0374e02501b2.png)


+ Batchsize-Accuracy curve with Deep KWS:

![acc_KWS](https://user-images.githubusercontent.com/63407850/158172962-eff34f16-26be-43e6-afa0-a908b8642cc2.png)


+ Accuracy: 

| Module   | Epoch1 | Epoch2 | epoch3 | epoch4 | epoch5 | text   |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| DNN      | 38.57% | 52.85% | 58.81% | 67.48% | 71.00% | 62.59% |
| CNN      | 95.30% | 96.12% | 96.30% | 97.20% | 96.75% | 95.17% |
| TDNN     | 70.10% | 69.02% | 74.35% | 77.87% | 80.76% | 76.50% |
| LSTM     | 57.36% | 74.35% | 75.16% | 79.31% | 81.39% | 78.75% |
| DFSMN    | 91.15% | 92.14% | 94.94% | 93.86% | 94.04% | 90.34% |
| DNN(KWS) | 87.97% | 90.37% | 91.04% | 91.18% | 90.44% | 89.67% |

**Speech wake-up：**

+ Accuracy with Deep_KWS:

![acc](https://user-images.githubusercontent.com/63407850/158172843-73ad9507-d7fe-4087-898a-4afbec1d3278.png)

+ Loss with Deep_KWS:

![loss](https://user-images.githubusercontent.com/63407850/158172839-e1a75c50-bfed-485c-9e69-d552d96e05f0.png)

