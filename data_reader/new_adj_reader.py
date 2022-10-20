import numpy as np
import torch

def cell_adj(cell, size):
    new_indes = np.loadtxt(cell)
    new_indes = torch.from_numpy(new_indes).long()
    new_indes = new_indes.transpose(0, 1)
    train1 = []
    val1 = []
    test1 = []
    a = int(size * 0.8)
    c = int(size * 0.1)

    for i in range(0, size):
        if (i < a):
            train1.append(i)
        elif (i > (a+c) and i < size):
            test1.append(i)
        else:
            val1.append(i)
    train1 = np.array(train1)
    train1 = torch.from_numpy(train1).long()
    val1 = np.array(val1)
    val1 = torch.from_numpy(val1).long()
    test1 = np.array(test1)
    test1 = torch.from_numpy(test1).long()
    return new_indes, train1, val1, test1
