import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y = list(range(1, 501))
    y = np.asarray(y)
    y = y * 10

    print(1)
    r1 = np.load('results/1/result.npy')
    r2 = np.load('results/2/result.npy')
    r3 = np.load('results/3/result.npy')
    r4 = np.load('results/4/result.npy')
    r5 = np.load('results/5/result.npy')
    print(2)

    plt.plot(y, r5)
    plt.axhline(y=0, color='r')
    plt.show()

