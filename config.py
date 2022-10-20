import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import logging
warnings.filterwarnings("ignore")
import time


class Params():
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()

def config():
    cfg = Params()
    cfg.dirData = 'D:/PycharmWorkplace/MATGNN/scDatasets/'
    # BaronMouse 13 1886 BaronHuman 14 8569  Muraro 10 2122 Segerstolpe 10 2133 Xin10 4 1449 Zhengsorted 10 20000
    cfg.dataset = 'Muraro'
    cfg.cellAdj = 'D:/PycharmWorkplace/MATGNN/celladj/Muraro_n5.txt'
    cfg.dirAdj = 'D:/PycharmWorkplace/MATGNN/scDatasets/Muraro/'
    cfg.dirLabel = 'D:/PycharmWorkplace/MATGNN/scDatasets/'
    cfg.outputDir = 'D:/PycharmWorkplace/MATGNN/data/output'
    cfg.saveResults = 0
    cfg.roc_directory = 'roc_result/macro'
    cfg.normalized_laplacian = True
    cfg.num_gene = 1000
    cfg.epochs = 100
    cfg.batchsize = 64
    cfg.dropout = 0.1
    cfg.net = 'String'
    cfg.dist = ''
    cfg.sampling_rate = 1
    cfg.drop_feature_rate_1 = 0.1
    cfg.drop_feature_rate_2 = 0.0
    cfg.confusion_directory = 'confusion'
    cfg.umap_directory2 = 'ump/gcn'

    cfg.layers       = 2               # gnn layers
    cfg.h_size       = 256              # hidden_size
    cfg.lr           = 0.0005           # learning rate
    cfg.drop_rate    = 0.2             # drop out
    cfg.epoch        = 800             # training epoches
    cfg.bias         = True            # whether using bias
    cfg.cuda         = '0'             # GPU ID
    cfg.T            = 2.0             # task weight soft
    cfg.gnn          = 'gcn'           # gnn backbone
    # cfg.task         = 'nlp'
    cfg.task         = 'nlp'           # n:node cls l: link pred p:pairwise
    cfg.act          = F.relu          # activation function
    cfg.split        = [166,83]        # 1000 for training 500 for validation; the rest nodes are used for testing 1000 用于训练 500 用于验证； 其余节点用于测试
    cfg.ratio        = 0.5             # training ratio
    cfg.fea_size     = 1000
    cfg.num_class    = 9               # Xin:4 BaronHuman:14 BaronMouse:13 Muraro:9 Segerstolpe:10 Zhengsorted:10 Zheng68K:11
    cfg.patience     = 10              # early stopping
    cfg.neg_size     = 1               # neg:pos
    cfg.threshold    = 0.8             # cosine similarity threshold
    cfg.start_es     = 100             # start early stopping
    cfg.current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    cfg.dyn_w        = "1"             # dynamic weight
    cfg.heads        = 4               # multi-heads for GAT
    cfg.attn_drop    = 0.6             # attention droprate for GAT
    cfg.neg_slope    = 0.1             # negative_slope for LeakyReLu in GAT
    cfg.residual     = False           # residual for GAT
    cfg.agg_type     = 'gcn'           # Aggregator type: mean/gcn/pool/lstm
    cfg.mad_calc     = True            # calculate mad
    cfg.subtask_eval = True            # whether evaluate subtasks
    cfg.classifier   = ['lg']          # model for subtask
    cfg.alpha        = 1e-6            # hyper-parameter

    return cfg


def log_config(cfg,logger):
    for key, value in cfg.__dict__.items():
        logger.info('{} : {}'.format(key,value))

cfg = config()