

"""
Created on Tue Mar 23 18:30:20 2021

@author: tianyu
"""
import time
import pandas as pd
import torch

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

import numpy as np
import sys
sys.path.insert(0, 'lib/')
import lib.utilsdata


def calculation(pred_test, test_labels, method='GCN'):
    test_acc = metrics.accuracy_score(pred_test, test_labels)
    test_f1_macro = metrics.f1_score(pred_test, test_labels, average='macro')
    test_f1_micro = metrics.f1_score(pred_test, test_labels, average='micro')
    precision = metrics.precision_score(test_labels, pred_test, average='micro')
    recall = metrics.recall_score(test_labels, pred_test, average='micro')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, pred_test, pos_label=2)
    auc = metrics.auc(fpr, tpr)
        
    print('method','test_acc','f1_test_macro','f1_test_micro','Testprecision','Testrecall','Testauc')
    print(method, test_acc, test_f1_macro, test_f1_micro, precision,recall,auc )
        
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.0)

def test_model(net, loader, L, args):
    t_start_test = time.time()

    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([args.nclass, args.nclass])
    predictions = pd.DataFrame()
    y_true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = net(batch_x, L)
        
        test_acc += lib.utilsdata.accuracy(pred, batch_y).item() * len(batch_y)
        count += 1
        y_true = batch_y.detach().cpu().numpy()
        y_predProbs = pred.detach().cpu().numpy()
        
    predictions = pd.DataFrame(y_predProbs)            
    for i in range(len(y_true)):
        confusionGCN[y_true[i], np.argmax(y_predProbs[i,:])] += 1
    
    t_total_test = time.time() - t_start_test
    preds_labels = np.argmax(np.asarray(predictions), 1)
    test_acc = test_acc/len(loader.dataset)
    predictions.insert(0, 'trueLabels', y_true)

    
    return test_acc, confusionGCN, predictions, preds_labels, t_total_test

        
def train_model(net, train_loader, val_loader, L, args):















    l2_regularization = 5e-4
    batch_size = args.batchsize
    num_epochs = args.epochs
    

    nb_iter = int(num_epochs * args.train_size) // batch_size
    print('num_epochs=', num_epochs,', train_size=', args.train_size,', nb_iter=',nb_iter)
    

    global_lr = args.lr
    global_step = 0
    decay = 0.95
    decay_steps = args.train_size
        
        



    
    if torch.cuda.is_available():
        net.cuda()
        
    print(net)
            

    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr= args.lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    

    net.train()
    losses_train = []
    acc_train = []
    
    t_total_train = time.time()

    def adjust_learning_rate(optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        lr = lr * pow( decay , float(global_step// decay_steps) )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    for epoch in range(num_epochs):
    

        cur_lr = adjust_learning_rate(optimizer,epoch, args.lr)
        

        t_start = time.time()
    

        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
            optimizer.zero_grad()

            output = net(batch_x, L)

            loss_batch = net.loss(output, batch_y, l2_regularization)
          
            acc_batch = lib.utilsdata.accuracy(output, batch_y).item()
            
            loss_batch.backward()
            optimizer.step()
            
            count += 1
            epoch_loss += loss_batch.item()
            epoch_acc += acc_batch
            global_step += args.batchsize 
            

            if count % 1000 == 0:
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))
    
    
        epoch_loss /= count
        epoch_acc /= count
        losses_train.append(epoch_loss)
        acc_train.append(epoch_acc)

        t_stop = time.time() - t_start
        
        if epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                val_acc = 0  
                count = 0
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(device), b_y.to(device)          
                    val_pred = net(b_x, L)
                    val_acc += lib.utilsdata.accuracy(val_pred, b_y).item() * len(b_y)
                    count += 1
                    
                val_acc = val_acc/len(val_loader.dataset)
                
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
            print('----accuracy(val)= ', val_acc)
            print('training_time:',t_stop)
        else:
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
            print('training_time:',t_stop)
        
    
    t_total_train = time.time() - t_total_train  
    
    return net, t_total_train

