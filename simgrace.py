from tqdm import *
import argparse
from  model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch_sparse import SparseTensor
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
from torch import optim
from torch.nn.parameter import Parameter
from model import *
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F



def evaluate(args, data, model):
    topk = 10000
    device = args.device
    data.edge_index, data.edge_attr = convert_to_directed_edge(data.edge_index, data.edge_attr)

    data = train_test_split_edges(data, val_ratio=0, test_ratio=0.2)
    model = model.to(device)
    data = data.to(device)
    optimizer_gcn = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    best_loss = 999999
    
    model.eval()
    embeddings = model(data.x, data.train_pos_edge_index, data.train_pos_edge_attr)
    # link prediction task!
    all_edges_test = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=1).to(device)
    num_positive_edges_test = data.test_pos_edge_index.shape[1]
    edge_scores_test = model.decoder(embeddings, all_edges_test)
    labels_test = torch.cat([torch.ones(num_positive_edges_test), torch.zeros(num_positive_edges_test)])
    auc_test = roc_auc_score(labels_test.cpu().detach().numpy(), edge_scores_test.cpu().detach().numpy())
    print(f'Test AUC: {auc_test}')

    
    # predict this masked edges
    exist_edges = data.train_pos_edge_index.T.cpu().detach().numpy().tolist()
    masked_edges = data.test_pos_edge_index.T.cpu().detach().numpy().tolist()
    exist_edges_set = set()
    masked_edges_set = set()
    pred_edges_set = set()
    
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    similarity_matrix = similarity_matrix.triu(diagonal=1)  # 只保留上三角矩阵
    node_nums = embeddings.shape[0]
    score, indices = torch.topk(similarity_matrix.view(-1), k=topk*10)
    row_indices = indices // node_nums
    col_indices = indices % node_nums
    pred_edges = torch.stack([row_indices, col_indices], dim=0).T.cpu().detach().numpy().tolist()

    for i in exist_edges:
        exist_edges_set.add(tuple(sorted([i[0],i[1]])))
    for i in masked_edges:
        masked_edges_set.add(tuple(sorted([i[0],i[1]])))
    for i in pred_edges:
        pred_edges_set.add(tuple(sorted([i[0],i[1]])))
    pred_edges_set = set(list(pred_edges_set - exist_edges_set)[0:topk])
    corr_nums = len(pred_edges_set&masked_edges_set)
    precision = corr_nums / topk

    return round(auc_test,4), precision


class GNN_Backbone(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers = 2, gnn = 'gcn'):
        super(GNN_Backbone, self).__init__()
        self.embedding_layer = nn.Embedding(30000, hid_dim)
        self.lin_layer = nn.Linear(in_dim, hid_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            if gnn == 'gcn':
                self.convs.append(GCNConv(hid_dim, hid_dim))
            elif gnn == 'gcn-new':
                self.convs.append(GCN_Conv(hid_dim, hid_dim))
            elif gnn == 'gin':
                self.convs.append(GIN_Conv(hid_dim, hid_dim))
            elif gnn == 'gat':
                self.convs.append(GATConv(hid_dim, hid_dim))
            elif gnn == 'sage':
                self.convs.append(SAGEConv(hid_dim, hid_dim))
            elif gnn == 'gatv2':
                self.convs.append(GATv2Conv(hid_dim, hid_dim))
            # self.norms.append(nn.BatchNorm1d(hid_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.edge_encoder = nn.Linear(1, hid_dim)
        self.classifer = nn.Linear(hid_dim, 1)
        self.drop_ratio = 0.5

    def agg_edge(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr.unsqueeze(1))
        src, dst = edge_index
        new_x = x.clone()
        new_x[src] = F.relu(x[src] + edge_embedding)
        new_x[dst] = F.relu(x[dst] + edge_embedding)
        return new_x

    def forward(self, x, edge_index, edge_attr=None):
        if x.shape[1] == 1:
            x = self.embedding_layer(x.squeeze().long())
        else:
            x = self.lin_layer(x)
        if edge_attr != None:
            x = self.agg_edge(x, edge_index, edge_attr.squeeze())
        h = x
        h_list = []
        for layer in range(self.num_layers):

            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
        return torch.cat(h_list, 1)

    def decoder(self, z, edge_index):
        edge_scores = torch.sigmoid(self.classifer(z[edge_index[0]] * z[edge_index[1]]))
        return edge_scores.squeeze()

class simclr(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, gnn = 'gcn'):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = GNN_Backbone(in_dim, hidden_dim, num_gc_layers, gnn)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()
        self.classifer = nn.Linear(self.embedding_dim, 1)

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, edge_attr):
        edge_attr = None
        x = self.encoder(x, edge_index, edge_attr)
        x = self.proj_head(x)        
        return x

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def decoder(self, z, edge_index):
        edge_scores = torch.sigmoid(self.classifer(z[edge_index[0]] * z[edge_index[1]]))
        return edge_scores.squeeze()

def gen_ran_output(data, model, vice_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if param.data.std() > 0 and adv_param.data.std():
            if name.split('.')[0] == 'proj_head':
                adv_param.data = param.data
            else:
                adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(args.device)           
    vice_model = vice_model.to(args.device)
    z2 = vice_model(data.x, data.edge_index, data.edge_attr)
    return z2

def main(args, data, model, vice_model):

    device = args.device
    data.edge_index, data.edge_attr = convert_to_directed_edge(data.edge_index, data.edge_attr)
    num_node = data.x.shape[0]
    in_dim = data.x.shape[1]
    hid_dim = 32
    best_auc = 0

    # obtain positive edges and negative
    data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.1)
    print(data.train_pos_edge_attr.shape)
    model = model.to(device)
    data = data.to(device)
    optimizer_gcn = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        optimizer_gcn.zero_grad()

        # ============ simgrace loss ===================
        x2 = gen_ran_output(data, model, vice_model, args)
        x1 = model(data.x, data.edge_index, data.edge_attr)
        loss_simgrace = model.loss_cal(x2, x1)
        # ============ simgrace loss ===================

        # ============ link prediction loss ===================
        # obtain gene embedding
        if args.edge_attr == 1:
            embeddings = model(data.x, data.train_pos_edge_index, data.train_pos_edge_attr)
        else:
            embeddings = model(data.x, data.train_pos_edge_index)

        # build training edges
        all_edges = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=1)
        num_positive_edges = data.val_pos_edge_index.shape[1]

        # obtain prediction edge score for validation set
        edge_scores_val = model.decoder(embeddings, all_edges)
        labels_val = torch.cat([torch.ones(num_positive_edges), torch.zeros(num_positive_edges)]).to(device)
        loss_lp = criterion(edge_scores_val, labels_val)

        # ============ link prediction loss ===================

        all_loss =  loss_simgrace + loss_lp
        all_loss.backward()
        optimizer_gcn.step()


        # print('Epoch {}, Simgrace Loss {}, Link Loss {}'.format(epoch, loss_simgrace.item(), loss_lp.item()))
    
        # Evaluation on the test set
        if epoch % 5 == 0:
            model.eval()

            # Build test edges
            all_edges_test = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=1).to(device)
            num_positive_edges_test = data.test_pos_edge_index.shape[1]

            # Obtain prediction edge score for the test set
            edge_scores_test = model.decoder(embeddings, all_edges_test)
            labels_test = torch.cat([torch.ones(num_positive_edges_test), torch.zeros(num_positive_edges_test)])
            # Calculate AUC for the test set
            auc_test = roc_auc_score(labels_test.cpu().detach().numpy(), edge_scores_test.cpu().detach().numpy())
            print(f'Test AUC: {auc_test}')
            
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn',
                        help='gcn, gat, gatv2, sage, vgae')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--hid_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='max training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--iter', type=int, default=1,
                        help='# training cell graphs')
    parser.add_argument('--feat', type=str, default='s-anno',
                        help='id, anno, cancer')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='device')
    parser.add_argument('--edge_attr', type=int, default=1,
                        help='edge attribution')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='eta')
    
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    df = pd.read_csv('/data7/chenyihan/katrina/ppi-experiments/processed_data/mapping_data/cell_name_shuffled.txt')
    df.columns = ['cell_name']
    all_cell_name = df['cell_name'].values.tolist()
    in_dim = args.hid_dim
    if args.feat == 'id':
        in_dim = 32
    elif 'g' in args.feat:
        in_dim = 4096
    elif args.feat == 's-anno':
        in_dim = 768
    else:
        in_dim = 768 + 4096
    # if args.model in ['gcn', 'gat', 'gatv2', 'sage']:
    #     model = GNN_Backbone(in_dim=in_dim, hid_dim=args.hid_dim, num_layers = args.num_layers, gnn = args.model)
    # elif args.model == 'vgae':
    
    model = simclr(in_dim, args.hid_dim, args.num_layers, gnn=args.model)
    vice_model = simclr(in_dim, args.hid_dim, args.num_layers, gnn=args.model)

    lens = 0
    old_cell_line_name = ''
    for cell in tqdm(all_cell_name[0:30001]):
        data = get_data(cell, args)
        model = main(args, data, model, vice_model)
        evaluate_data = get_data(all_cell_name[-1], args)
        auc, precision = evaluate(args, evaluate_data, model)
        print('auc:{}, precision:{}'.format(auc, precision))
        
        lens += 1
        if  lens in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]:
            torch.save(model, f'./checkpoints/simgrace_{args.feat}_{lens}.pth')
        if lens == 10:
            break