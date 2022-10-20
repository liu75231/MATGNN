import numpy as np
import torch
import random
import torch.utils.data as Data

def cell_adj(cell, size):


    f = open(cell, 'rb')
    new_indes = np.fromfile(f,dtype='<f')
    new_indes = new_indes.reshape((size, size))
    return new_indes


def generate_loader(train_data, train_labels):

    dset_train = Data.TensorDataset(train_data, train_labels)
    train_loader = Data.DataLoader(dset_train, batch_size = 20000, shuffle=True)
    print()
    return train_loader











def rom_work(adj):


    h = adj.shape[0]
    drop_h = int(0.025 * h)
    a = 0

    step1 = 0
    step2 = 0
    step3 = 0
    step4 = 0

    x = random.randint(0, int(0.5*h-2))
    y = random.randint(0, int(0.5*h-2))
    while a < drop_h:
        if(x < 0.5*h and y < 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step1 +=1
            if(adj[x][y] == 1):
                a+=1
        else:


            x = random.randint(0, int(0.5 * h - 2))
            y = random.randint(0, int(0.5 * h - 2))

    x = random.randint(int(0.5 * h+2), h-2)
    y = random.randint(0, int(0.5 * h-2))
    a = 0
    while a < drop_h:
        if(x < h-1 and y < 0.5* h and x >= 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step2 += 1
            if(adj[x][y] == 1):
                a+=1
        else:


            x = random.randint(int(0.5 * h+2), h - 2)
            y = random.randint(0, int(0.5 * h-2))

    y = random.randint(int(0.5 * h+2), h - 2)
    x = random.randint(0, int(0.5 * h-2))
    a = 0
    while a < drop_h:
        if(y < h-1 and x < 0.5* h and y >= 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step3 += 1
            if(adj[x][y] == 1):
                a+=1
        else:


            y = random.randint(int(0.5 * h+2), h - 2)
            x = random.randint(0, int(0.5 * h-2))



    x = random.randint(int(0.5 * h+2), h - 2)
    y = random.randint(int(0.5 * h+2), h - 2)
    a = 0
    while a < drop_h:
        if(y < h-1 and x >= 0.5* h and y >= 0.5*h and x < h-1):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step4 += 1
            if(adj[x][y] == 1):
                a+=1
        else:


            x = random.randint(int(0.5 * h+2), h - 2)
            y = random.randint(int(0.5 * h+2), h - 2)
    suml = step1+step2+step3+step4
    avg = int((step1+step2+step3+step4)/4)
    ld_1 = 0.1*(1-step1/suml)*h
    ld_2 = 0.1*(1-step2/suml)*h
    ld_3 = 0.1*(1-step3/suml)*h
    ld_4 = 0.1*(1-step4/suml)*h
    x = random.randint(0, int(0.5 * h+2))
    y = random.randint(0, int(0.5 * h+2))
    while a < ld_1:
        if(x < 0.5*h and y < 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step1 +=1
            if(adj[x][y] == 1):
                adj[x][y] == 0
                a+=1
        else:


            x = random.randint(0, int(0.5 * h-2))
            y = random.randint(0, int(0.5 * h-2))



    x = random.randint(int(0.5 * h+2), h - 2)
    y = random.randint(0, int(0.5 * h+2))
    a = 0
    while a < ld_2:
        if(x < h-1 and y < 0.5* h and x >= 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step2 += 1
            if(adj[x][y] == 1):
                adj[x][y] == 0
                a+=1
        else:


            x = random.randint(int(0.5 * h+2), h - 2)
            y = random.randint(0, int(0.5 * h - 2))



    y = random.randint(int(0.5 * h+2), h - 2)
    x = random.randint(0, int(0.5 * h - 2))
    a = 0
    while a < ld_3:
        if(y < h-1 and x < 0.5* h and y >= 0.5*h):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step3 += 1
            if(adj[x][y] == 1):
                adj[x][y] == 0
                a+=1
        else:


            y = random.randint(int(0.5 * h+2), h - 2)
            x = random.randint(0, int(0.5 * h - 2))



    x = random.randint(int(0.5 * h+2), h - 2)
    y = random.randint(int(0.5 * h+2), h - 2)
    a = 0
    while a < ld_4:
        if(y < h-1 and x >= 0.5* h and y >= 0.5*h and x < h-1):
            step_hor = 1 if random.randint(0, 1) else -1
            step_ver = 1 if random.randint(0, 1) else -1
            x = x+step_hor
            y = y+step_ver
            step4 += 1
            if(adj[x][y] == 1):
                adj[x][y] == 0
                a+=1
        else:


            x = random.randint(int(0.5 * h+2), h - 2)
            y = random.randint(int(0.5 * h+2), h - 2)
    return adj

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x



























