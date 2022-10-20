import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, JumpingKnowledge
import torch
from dgl.nn.pytorch.conv import GraphConv

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, 'lib/')


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from lib.coarsening import lmax_L
from lib.coarsening import rescale_L
from lib.utilsdata import sparse_mx_to_torch_sparse_tensor
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=att_dropout, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def graph_conv_cheby(self, x, cl,L, Fout, K):







        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)


        lmax = lmax_L(L)
        L = rescale_L(L, lmax)


        L = sparse_mx_to_torch_sparse_tensor(L)

        if torch.cuda.is_available():
            L = L.cuda()


        x0 = x.permute(1,2,0).contiguous()
        x0 = x0.view([V, Fin*B])
        x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return torch.cat((x, x_), 0)

        if K > 1:
            x1 = torch.sparse.mm(L,x0)
            x = torch.cat((x, x1.unsqueeze(0)),0)
        for k in range(2, K):
            x2 = 2 * torch.sparse.mm(L,x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])
        x = x.permute(3,1,2,0).contiguous()
        x = x.view([B*V, Fin*K])


        x = cl(x)
        x = x.view([B, V, Fout])

        return x

    def forward(self, x, L):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x.log_softmax(dim=-1)


class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K, alpha,
        dropout):
        super(APPNPNet, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.prop = APPNP(K, alpha)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        x = self.prop(x, adj_t)
        return x.log_softmax(dim=-1)


class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='concat'):
        super().__init__()
        self.CL1_K = 5
        self.CL1_F = 5
        self.cl1 = nn.Linear(5, 5)
        self.convs = torch.nn.ModuleList()
        self.fc1 = nn.Linear(625, 256)
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def graph_conv_cheby(self, x, cl,L, Fout, K):







        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)


        lmax = lmax_L(L)
        L = rescale_L(L, lmax)


        L = sparse_mx_to_torch_sparse_tensor(L)

        if torch.cuda.is_available():
            L = L.cuda()


        x0 = x.permute(1,2,0).contiguous()
        x0 = x0.view([V, Fin*B])
        x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return torch.cat((x, x_), 0)

        if K > 1:
            x1 = torch.sparse.mm(L,x0)
            x = torch.cat((x, x1.unsqueeze(0)),0)
        for k in range(2, K):
            x2 = 2 * torch.sparse.mm(L,x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])
        x = x.permute(3,1,2,0).contiguous()
        x = x.view([B*V, Fin*K])


        x = cl(x)
        x = x.view([B, V, Fout])

        return x
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()
            x = nn.MaxPool1d(p)(x)
            x = x.permute(0,2,1).contiguous()
            return x
        else:
            return x

    def forward(self, x, L):
        xs = []
        x = x.unsqueeze(2)
        x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 8)
        x_l = x.view(-1, 625)
        x_l = self.fc1(x_l)
        for i, conv in enumerate(self.convs):
            x_l = self.bns[i](x_l)
            x_l = F.relu(x_l)
            x_l = F.dropout(x_l, p=self.dropout, training=self.training)
            xs += [x_l]

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def loss(self, y2, y_target2, l2_regularization):


        loss2 = nn.NLLLoss()(y2, y_target2)
        loss = 1 * loss2

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()


        loss += 0.2 * l2_regularization * l2_loss

        return loss
class JKNet1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='concat'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels, activation=F.relu))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, activation=F.relu))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, adj_t, x, calc_mad=False):
        xs = []
        outputs = []
        for i, conv in enumerate(self.convs):
            x = conv(adj_t, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]
            if calc_mad:
                outputs.append(x)

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)


        return x, outputs

class AKPGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(AKPGCN, self).__init__()
        self.CL1_K = 5
        self.CL1_F = 5
        self.cl1 = nn.Linear(5, 5)
        self.fc1 = nn.Linear(625, 256)
        self.fc2 = nn.Linear(256, 13)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
    def graph_conv_cheby(self, x, cl,L, Fout, K):







        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)


        lmax = lmax_L(L)
        L = rescale_L(L, lmax)


        L = sparse_mx_to_torch_sparse_tensor(L)

        if torch.cuda.is_available():
            L = L.cuda()


        x0 = x.permute(1,2,0).contiguous()
        x0 = x0.view([V, Fin*B])
        x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return torch.cat((x, x_), 0)

        if K > 1:
            x1 = torch.sparse.mm(L,x0)
            x = torch.cat((x, x1.unsqueeze(0)),0)
        for k in range(2, K):
            x2 = 2 * torch.sparse.mm(L,x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])
        x = x.permute(3,1,2,0).contiguous()
        x = x.view([B*V, Fin*K])


        x = cl(x)
        x = x.view([B, V, Fout])

        return x
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()
            x = nn.MaxPool1d(p)(x)
            x = x.permute(0,2,1).contiguous()
            return x
        else:
            return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, L):
        xs = []
        x = x.unsqueeze(2)
        x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 8)
        x_l = x.view(-1, 625)
        x_l = self.fc1(x_l)
        for i, conv in enumerate(self.convs[:-1]):

            x_l = self.bns[i](x_l)
            x_l = F.relu(x_l)
            x_l = F.dropout(x_l, p=self.dropout, training=self.training)

        x_l =self.fc2(x_l)
        return x_l
    def loss(self, y2, y_target2, l2_regularization):


        loss2 = nn.NLLLoss()(y2, y_target2)
        loss = 1 * loss2

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()


        loss += 0.2 * l2_regularization * l2_loss

        return loss