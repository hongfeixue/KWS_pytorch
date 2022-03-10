import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

def plot_loss(n):
    y = []
    for i in range(1,n):
        enc = torch.load('.\loss\epoch_{}'.format(i))
        tempy = list(enc)
        y += tempy
    x = range(0,len(y))
    y = savgol_filter(y, 99, 1, mode='nearest')
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 100; LEARNING_RATE:0.005'
    plt.title(plt_title)
    plt.xlabel('Batch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    plot_loss(2)