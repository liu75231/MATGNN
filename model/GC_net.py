from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
import torch
from torch_geometric.nn.conv import GraphConv as conv, ChebConv
from torch.functional import F
from math import floor
from lib.coarsening import lmax_L
from lib.coarsening import rescale_L
from lib.utilsdata import sparse_mx_to_torch_sparse_tensor
from lib.utilsdata import sparse_mx_to_indices
import numpy as np
import torch.nn as nn
class base_GC_net(torch.nn.Module):

    def __init__(self, in_channels, n_gc_hidden_units, n_class,n_layer=0, k=0,drop_prob=0, conv_act=lambda x: x, output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k=k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()



        self.conv_layers.append(conv(self.in_channels, self.out_channels))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(conv(self.out_channels, self.out_channels).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))

        self.aggr_norm = torch.nn.BatchNorm1d(n_layer * self.out_channels)



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * n_layer * 3)


        self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, self.out_channels * n_layer * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * n_layer * 2, self.out_channels * n_layer)
        self.lin3 = torch.nn.Linear(self.out_channels * n_layer, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, floor(self.out_channels / 2) * n_layer)

            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * n_layer, self.n_class)

        self.reset_parameters()


    def reset_parameters(self):
        for gc_layer, batch_norm in zip(self.conv_layers, self.gc_layer_norm):
            gc_layer.reset_parameters()
            batch_norm.reset_parameters()

        self.aggr_norm.reset_parameters()

        self.aggr_norm.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


    def forward(self, data, L, batch, device):


        L = L.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((L.row, L.col)).astype(np.int64))
        edge_index = L
        m = data[0]
        new_x = m.float()
        X = new_x.view(100, 1)
        data_size = data.size(0)
        numly = data_size * 100
        n = np.zeros(numly)
















        if (data_size == 2):
            for i in range(1, data_size):
                m = data[i]
                x = m.float()
                x = x.view(100, 1)

                X = torch.cat((X, x), 0)
                print(X)
                j = torch.from_numpy(
                    np.vstack((L.row + i * 100, L.col + i * 100)).astype(np.int64))
                edge_index = torch.cat((indices, j), 1).to(device)
        else:
            n = np.zeros(numly)
            j = 0
            for i in range(0, numly):

                if (i != 0):
                    if (i % 100 == 0):
                        j += 1
                n[i] = j
            batch = torch.from_numpy(n).long().to(device)
            for i in range(1, data_size):
                m = data[i]
                x = m.float()
                x = x.view(100, 1)

                X = torch.cat((X, x), 0)
                j = torch.from_numpy(
                    np.vstack((L.row + i * 100, L.col + i * 100)).astype(np.int64))
                edge_index = torch.cat((indices, j), 1).to(device)
        h = X
        H = []
        for gc_layer, batch_norm in zip(self.conv_layers, self.gc_layer_norm):
            h = gc_layer(h, edge_index)
            h = self.conv_act(h)
            h = batch_norm(h)

            H.append(h)

        H = torch.cat(H, dim=1)
        H_avg=gap(H, batch)
        H_add=gadd(H, batch)
        H_max=gmp(H, batch)

        H = torch.cat([H_avg, H_add, H_max], dim=1)



        if self.output == "funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        else:
            assert False, "error in the read output"

    def funnel_output(self, H):
        '''
        readout part composed of a sequence of layer with dimension m*2, m, n_class, respectively
        :param H: the graph layer representation computed by the PGC-layer
        :return: the output of the model
        '''

        x = self.bn_out(H)

        x = F.relu(self.lin1(x))

        x = self.dropout(x)

        x = F.relu(self.lin2(x))
        x = self.dropout(x)

        x = self.out_fun(self.lin3(x))

        return x

    def restricted_funnel_output(self, H):
        '''
        readout part composed of a sequence of layers with dimension m/2, n_class, respectively
        :param H: the graph layer representation computed by the PGC-layer
        :return: the output of the model
        '''
        x = self.bn_out(H)

        x = self.dropout(x)

        x = F.relu(self.lin1(x))

        x = self.dropout(x)

        x = self.out_fun(self.lin2(x))

        return x

    def loss(self,  y2, y_target2, l2_regularization):


        loss2 = nn.NLLLoss()(y2, y_target2)
        loss = 1 * loss2

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()


        loss += 0.2 * l2_regularization * l2_loss

        return loss

class Cheb_net(base_GC_net):

    def __init__(self, in_channels, n_gc_hidden_units, n_class, n_layer=0, k=0, drop_prob=0, conv_act=lambda x: x,
                 output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k = k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()


        self.conv_layers.append(ChebConv(self.in_channels, self.out_channels, k))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(ChebConv(self.out_channels, self.out_channels, k).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))

        self.aggr_norm = torch.nn.BatchNorm1d(n_layer * self.out_channels)



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * n_layer * 3)


        self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, self.out_channels * n_layer * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * n_layer * 2, self.out_channels * n_layer)
        self.lin3 = torch.nn.Linear(self.out_channels * n_layer, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, floor(self.out_channels / 2) * n_layer)

            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * n_layer, self.n_class)

        self.reset_parameters()


from torch_geometric.nn import GCNConv, JumpingKnowledge
class JK_net(base_GC_net):

    def __init__(self, in_channels, n_gc_hidden_units, n_class, n_layer=0, k=0, drop_prob=0, conv_act=lambda x: x,
                 output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k = k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()


        self.conv_layers.append(GCNConv(self.in_channels, self.out_channels))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(GCNConv(self.out_channels, self.out_channels).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))

        self.aggr_norm = torch.nn.BatchNorm1d(n_layer * self.out_channels)

        self.jk_layer = JumpingKnowledge(mode="cat")



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * n_layer *3 )


        self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, self.out_channels * n_layer * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * n_layer * 2, self.out_channels * n_layer)
        self.lin3 = torch.nn.Linear(self.out_channels * n_layer, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, floor(self.out_channels / 2) * n_layer)

            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * n_layer, self.n_class)

        self.reset_parameters()

    def loss(self, y2, y_target2, l2_regularization):


        loss2 = nn.NLLLoss()(y2, y_target2)

        loss = 1 * loss2
        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()


        loss += 0.2 * l2_regularization * l2_loss

        return loss

    def forward(self, data, L, batch, device):










        L = L.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((L.row, L.col)).astype(np.int64))
        edge_index = L
        m = data[0]
        new_x = m.float()
        X = new_x.view(100, 1)
        data_size = data.size(0)
        numly = data_size * 100
        n = np.zeros(numly)
















        if(data_size == 2):
            for i in range(1, data_size):
                m = data[i]
                x = m.float()
                x = x.view(100, 1)

                X = torch.cat((X, x), 0)
                print(X)
                j = torch.from_numpy(
                            np.vstack((L.row+i*100, L.col+i*100)).astype(np.int64))
                edge_index = torch.cat((indices, j), 1).to(device)
        else:
            n = np.zeros(numly)
            j = 0
            for i in range(0, numly):

                if (i != 0):
                    if (i % 100 == 0):
                        j += 1
                n[i] = j
            batch = torch.from_numpy(n).long().to(device)
            for i in range(1, data_size):
                m = data[i]
                x = m.float()
                x = x.view(100, 1)

                X = torch.cat((X, x), 0)
                j = torch.from_numpy(
                            np.vstack((L.row+i*100, L.col+i*100)).astype(np.int64))
                edge_index = torch.cat((indices, j), 1).to(device)

        h = X
        H = []
        for gc_layer, batch_norm in zip(self.conv_layers, self.gc_layer_norm):

            h = gc_layer(h, edge_index)
            h = self.conv_act(h)
            h = batch_norm(h)

            H.append(h)

        H = self.jk_layer(H)
        H_avg = gap(H, batch)
        H_add = gadd(H, batch)
        H_max = gmp(H, batch)
        H = torch.cat([H_avg, H_add, H_max], dim=1)




        if self.output == "funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        else:
            assert False, "error in the read output"


from torch_geometric.nn.conv import SGConv

class SGC_net(base_GC_net):
    def __init__(self, in_channels, n_gc_hidden_units, n_class,n_layer=0, k=0,drop_prob=0, conv_act=lambda x: x, output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k=k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()



        self.conv_layers.append(SGConv(self.in_channels, self.out_channels))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(SGConv(self.out_channels, self.out_channels).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))

        self.aggr_norm = torch.nn.BatchNorm1d(n_layer * self.out_channels)



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * n_layer * 3)


        self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, self.out_channels * n_layer * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * n_layer * 2, self.out_channels * n_layer)
        self.lin3 = torch.nn.Linear(self.out_channels * n_layer, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, floor(self.out_channels / 2) * n_layer)

            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * n_layer, self.n_class)

        self.reset_parameters()


from torch_geometric.nn.conv import APPNP

class APPNP_net(base_GC_net):

    def __init__(self, in_channels, n_gc_hidden_units, n_class,n_layer=0, k=0,drop_prob=0, conv_act=lambda x: x, output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k=k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()



        self.conv_layers.append(APPNP(K=k,alpha=0.5))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(APPNP(K=k,alpha=0.5).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))

        self.aggr_norm = torch.nn.BatchNorm1d(n_layer * self.out_channels)



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * n_layer * 3)


        self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, self.out_channels * n_layer * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * n_layer * 2, self.out_channels * n_layer)
        self.lin3 = torch.nn.Linear(self.out_channels * n_layer, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * n_layer * 3, floor(self.out_channels / 2) * n_layer)

            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * n_layer, self.n_class)

        self.reset_parameters()

class SIGN_net(base_GC_net):

    def __init__(self, in_channels, n_gc_hidden_units, n_class,n_layer=0, k=0,drop_prob=0, conv_act=lambda x: x, output=None,
                 device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.in_channels = in_channels
        self.n_layer = n_layer
        self.k=k
        self.out_channels = n_gc_hidden_units
        self.n_class = n_class
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.output = output

        self.lins = torch.nn.ModuleList()
        for _ in range(k):
            self.lins.append(torch.nn.Linear(in_channels, n_gc_hidden_units))



        self.out_fun = torch.nn.LogSoftmax(dim=1)
        self.bn_hidden_rec = torch.nn.BatchNorm1d(self.out_channels * k)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * k * 3)


        self.lin1 = torch.nn.Linear(self.out_channels * k * 3, self.out_channels * k * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * k * 2, self.out_channels * k)
        self.lin3 = torch.nn.Linear(self.out_channels * k, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * k * 3, floor(self.out_channels / 2) * k)
            self.lin2 = torch.nn.Linear(floor(self.out_channels / 2) * k, self.n_class)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()


        self.bn_hidden_rec.reset_parameters()
        self.bn_out.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):

        H = [self.lins[0](data.x)]
        for i in range(1,self.k):
            h=self.lins[i](data[f'x{i}'])
            H.append(h)

        H = self.bn_hidden_rec(torch.cat(H, dim=1))

        H_avg = gap(H, data.batch)
        H_add = gadd(H, data.batch)
        H_max = gmp(H, data.batch)
        H = torch.cat([H_avg, H_add, H_max], dim=1)


        if self.output == "funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        else:
            assert False, "error in output stage"


