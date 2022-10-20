#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:00:37 2020

@author: tianyu
"""


from data_reader.new_adj_reader_xin import *
from config import cfg
from dgl import DGLGraph
import networkx as nx
import itertools as it
from sklearn.metrics import accuracy_score
from early_stop import EarlyStopping
import random
from numpy.linalg import inv
from gcn import GNN
from model.models import *
from sklearn.metrics import precision_score, recall_score, f1_score
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
from lib.coarsening import  coarsen, laplacian
import lib.utilsdata
from lib.utilsdata import *
from train import *
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")


class Task(nn.Module):
    def __init__(self, model):
        super(Task, self).__init__()
        self.model = model
        self.linear = nn.Linear(cfg.h_size, cfg.num_class).to(device)

    def forward(self, g, features, calc_mad):
        x, output = self.model(g, features, calc_mad)
        logits = self.linear(x)
        return x, logits, output

def dynamic_weight_average(loss_tracker, epoch):
    num_tasks = len(loss_tracker)
    weight_assign = []
    if epoch < 5:
        for k in range(num_tasks):
            weight_assign.append(1.0 / num_tasks)
        weight_assign = np.array(weight_assign)
    else:
        loss_sum = sum(loss_tracker)
        for k in range(num_tasks):
            weight_assign.append(loss_tracker[k] / loss_sum)
        weight_assign = np.array(weight_assign) / cfg.T
        max_w = np.max(weight_assign)
        weight_assign = weight_assign - max_w
        w_exp = np.exp(weight_assign)
        weight_assign = num_tasks * w_exp / w_exp.sum()
    return weight_assign


def main():
    t_start = time.process_time()
    print('load data...')
    adjall, alldata, labels, shuffle_index, reverse = load_largesc(path=cfg.dirData, dirAdj=cfg.dirAdj, dataset=cfg.dataset,
                                                          net='String')
    real_lable = list(reverse.values())
    if not (shuffle_index.all()):
        shuffle_index = shuffle_index.astype(np.int32)
    else:
        shuffle_index = np.random.permutation(alldata.shape[0])
        np.savetxt(cfg.dirData + '/' + cfg.dataset + '/shuffle_index_' + cfg.dataset + '.txt')

    train_all_data, adj = down_genes(alldata, adjall, cfg.num_gene)
    L = [laplacian(adj, normalized=True)]
    new_x = torch.from_numpy(train_all_data).float()
    x_1 = drop_feature(new_x, cfg.drop_feature_rate_1)
    x_2 = drop_feature(new_x, cfg.drop_feature_rate_2)
    new_indes = cell_adj(cfg.cellAdj, new_x.size(0))
    new_indes = rom_work(new_indes)
    try:
        print('Delete existing network\n')
    except NameError:
        print('No existing network to delete\n')

    dense_node_label = labels
    dense_node_label = dense_node_label.astype('int64')

    dense_adj = new_indes
    dense_adj = dense_adj.astype('int64')
    G_nx = nx.from_numpy_matrix(dense_adj, create_using=nx.Graph())
    G = DGLGraph(G_nx)
    G.add_edges(G.nodes(), G.nodes())
    dense_node_features = train_all_data
    dense_node_features = dense_node_features.astype('float32')
    train_nodes = []
    val_nodes = []
    test_nodes = []
    for i in range(cfg.num_class):
        x = np.argwhere(dense_node_label == i).reshape(-1)
        np.random.shuffle(x)
        size = x.shape[0]
        size1 = int(size * 0.8)
        size2 = int(size * 0.1)
        train_nodes.append(x[0:size1])
        test_nodes.append(x[size1:size1 + size2])
        val_nodes.append(x[size2 + size1:])
    pos_pairs = []
    for x in train_nodes:
        pos_pairs += list(it.combinations(x, 2))

    neg_pairs = []
    cross_labels = it.combinations(list(range(cfg.num_class)), 2)
    for i, j in cross_labels:
        x = train_nodes[i]
        y = train_nodes[j]
        neg_pairs += list(it.product(x, y, repeat=1))
    neg_pairs = random.sample(neg_pairs, cfg.neg_size * len(pos_pairs))

    train_pairs = np.array(pos_pairs + neg_pairs)
    train_pair_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))], 0)

    pos_pairs = []
    for x in val_nodes:
        pos_pairs += list(it.combinations(x, 2))

    neg_pairs = []

    cross_labels = it.combinations(list(range(cfg.num_class)), 2)
    for i, j in cross_labels:
        x = val_nodes[i]
        y = val_nodes[j]
        neg_pairs += list(it.product(x, y, repeat=1))
    neg_pairs = random.sample(neg_pairs, cfg.neg_size * len(pos_pairs))

    val_pairs = np.array(pos_pairs + neg_pairs)
    val_pair_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))], 0)
    train_nodes = np.concatenate(train_nodes, 0)
    val_nodes = np.concatenate(val_nodes, 0)
    test_nodes = np.concatenate(test_nodes, 0)
    y_train_true = dense_node_label[train_nodes]
    y_val_true = dense_node_label[val_nodes]
    y_test_true = dense_node_label[test_nodes]

    A = dense_adj
    A_tilde = A + np.identity(A.shape[0])
    D = A_tilde.sum(axis=1)

    Lambda = np.identity(A.shape[0])
    L = np.diag(D) - A
    P = inv(L + cfg.alpha * Lambda)

    train_dict = dict(zip(train_nodes, map(int, y_train_true)))
    probability = []
    for k in range(cfg.num_class):
        nodes = train_nodes[y_train_true == k]
        prob = P[:, nodes].sum(axis=1).flatten()
        probability.append(prob)

    probability = np.stack(probability, axis=1)
    probability = probability / np.sum(probability, axis=1, keepdims=True)
    probability = np.nan_to_num(probability)

    rw_predicted_labels = np.argmax(probability, axis=1)

    for node in train_nodes:
        rw_predicted_labels[node] = dense_node_label[node]
        probability[node][dense_node_label[node]] = 1.0

    pos_edges = []
    neg_edges = []
    for src, dst in list(G_nx.edges()):

        src_label = rw_predicted_labels[src]
        dst_label = rw_predicted_labels[dst]
        # if src_label!=dst_label:
        #     if probability[src][src_label]>cfg.threshold and probability[dst][dst_label]>cfg.threshold:
        #         continue
        if src_label == dst_label:
            if probability[src][src_label] > cfg.threshold and probability[dst][dst_label] > cfg.threshold:
                pos_edges.append([src, dst])

    if cfg.dataset != 'pubmed':
        candidate = list(nx.non_edges(G_nx))
        num_neg_edges = cfg.neg_size * len(pos_edges)
        neg_edges = []
        k = 0

        random.shuffle(candidate)
        for src, dst in candidate:
            src_label = rw_predicted_labels[src]
            dst_label = rw_predicted_labels[dst]
            if src_label != dst_label:
                if probability[src][src_label] > cfg.threshold and probability[dst][dst_label] > cfg.threshold:
                    neg_edges.append([src, dst])
                    k += 1
            if k == num_neg_edges:
                break
    else:
        neg_edges = []
        for src, dst in pos_edges:
            i = 0
            while i < cfg.neg_size:
                if random.random() > 0.5:

                    dst = random.randint(0, A.shape[0] - 1)
                else:
                    src = random.randint(0, A.shape[0] - 1)
                src_label = rw_predicted_labels[src]
                dst_label = rw_predicted_labels[dst]
                if src_label != dst_label:
                    if probability[src][src_label] > cfg.threshold and probability[dst][dst_label] > cfg.threshold:
                        neg_edges.append([src, dst])
                        i += 1

    edges = np.array(pos_edges + neg_edges)
    edge_labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))], 0)
    count = 0
    for i, (src, dst) in enumerate(pos_edges):
        if dense_node_label[src] == dense_node_label[dst]:
            x = 1
        else:
            x = 0
        if x == 1:
            count += 1

    print('true links : {:d} all links : {:d}'.format(count, len(pos_edges)))

    net = Task(GNN()).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    model = GNN().to(device)

    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

    train_loss = 0

    valid_loss = 0

    if cfg.task == "nlp":
        loss_tracker = [1.0, 1.0, 1.0]
    else:
        loss_tracker = [1.0]

    for i in range(cfg.epoch):
        net.train()
        x, y, _ = net(G.to(device), torch.FloatTensor(dense_node_features).to(device), False)
        train_logp = F.log_softmax(y[train_nodes], 1)
        n_loss = F.nll_loss(train_logp, torch.LongTensor(y_train_true).to(device))

        if "l" in cfg.task:
            edge_src = x[edges[:, 0]]
            edge_dst = x[edges[:, 1]]
            logits = torch.sum(torch.mul(edge_src, edge_dst), 1)
            l_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([cfg.neg_size]).to(device))(logits,
                                                                                                   torch.FloatTensor(
                                                                                                       edge_labels).to(
                                                                                                       device))

        if "p" in cfg.task:
            pair_src = y[train_pairs[:, 0]]
            pair_dst = y[train_pairs[:, 1]]
            logits = torch.sum(torch.mul(pair_src, pair_dst), 1)
            p_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([cfg.neg_size]).to(device))(logits,
                                                                                                   torch.FloatTensor(
                                                                                                       train_pair_labels).to(
                                                                                                       device))

        if cfg.dyn_w == "1":
            weight_assign = dynamic_weight_average(loss_tracker, i)
        else:
            weight_assign = loss_tracker

        if cfg.task == "nlp":
            loss = weight_assign[0] * n_loss + weight_assign[1] * l_loss + weight_assign[2] * p_loss
        elif cfg.task == "n":
            loss = n_loss

        train_n_loss = n_loss.item()
        if "l" in cfg.task:
            train_l_loss = l_loss.item()
        if "p" in cfg.task:
            train_p_loss = p_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_train_pred = np.argmax(train_logp.detach().cpu().numpy(), 1)
        train_acc = accuracy_score(y_train_true, y_train_pred)

        train_loss = n_loss.item()

        net.eval()
        with torch.no_grad():
            x, y, _ = net(G.to(device), torch.FloatTensor(dense_node_features).to(device), False)
            val_logp = F.log_softmax(y[val_nodes], 1)
            n_loss = F.nll_loss(val_logp, torch.LongTensor(y_val_true).to(device))

            if "l" in cfg.task:
                edge_src = x[edges[:, 0]]
                edge_dst = x[edges[:, 1]]
                logits = torch.sum(torch.mul(edge_src, edge_dst), 1)
                l_loss = nn.BCEWithLogitsLoss()(logits, torch.FloatTensor(edge_labels).to(device))

            if "p" in cfg.task:
                pair_src = y[val_pairs[:, 0]]
                pair_dst = y[val_pairs[:, 1]]
                logits = torch.sum(torch.mul(pair_src, pair_dst), 1)
                p_loss = nn.BCEWithLogitsLoss()(logits, torch.FloatTensor(val_pair_labels).to(device))

            if cfg.task != 'n':
                if cfg.dyn_w == "1":
                    for t in list(cfg.task):
                        if t == "n":
                            train_t_loss = train_n_loss
                            t_loss = n_loss.item()
                        elif t == "l":
                            train_t_loss = train_l_loss
                            t_loss = l_loss.item()
                        elif t == "p":
                            train_t_loss = train_p_loss
                            t_loss = p_loss.item()
                        loss_tracker.append((train_t_loss - t_loss) / train_t_loss)

            y_val_pred = np.argmax(val_logp.detach().cpu().numpy(), 1)
            val_acc = accuracy_score(y_val_true, y_val_pred)

            valid_loss = n_loss.item()

            if (i + 1) % 10 == 0:
                print(
                    'Epoch {:05d} | Train Loss {:.4f} Valid loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f}'.format(
                        i + 1, train_loss, valid_loss, train_acc, val_acc))


    net.eval()
    x, logits, output = net(G.to(device), torch.FloatTensor(dense_node_features).to(device), cfg.mad_calc)

    test_logp = F.softmax(logits[test_nodes], 1)
    newdl = test_logp.detach().cpu().numpy()
    y_test_pred = np.argmax(test_logp.detach().cpu().numpy(), 1)
    test_acc = accuracy_score(y_test_true, y_test_pred)

    print('Test accuracy_score = {:.4f}'.format(test_acc))
if __name__ == '__main__':
    main()






