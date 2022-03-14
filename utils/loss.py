from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F 
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Net = ["DNN","CNN","TDNN","LSTM","DFSMN"]
DNN = ["DNN", "DNNKWS"]

def plot_loss(n):
    for net in DNN:
        y = []
        x1 = []
        y1 = []
        for i in range(1,n):
            acc = torch.load('./accuary/'+ net + '_{}'.format(i))
            acc = list(acc)
            y += acc
        x = range(0,len(y))
        y = savgol_filter(y, 99, 1, mode='nearest')
        for i in range(0, len(x), 100):
            x1.append(x[i])
            y1.append(y[i])
        plt.plot(x, y, '.-',linewidth=2, label=net)
        # plt.plot(x1,y1,'o')

        plt.legend(loc=0, ncol=2)
        # plt_title = 'BATCH_SIZE = 100; LEARNING_RATE:0.005'
        # plt.title(plt_title)
        plt.xlabel('Batch')
        plt.ylabel('Accuary')
    # plt.show()
    plt.savefig('acc_KWS')

if __name__ == "__main__":
    plot_loss(5)