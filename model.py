import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import negative_sampling, to_undirected, train_test_split_edges
from utils import *
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import VGAE, InnerProductDecoder
from torch_geometric.nn import MessagePassing, GraphUNet, GATConv, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

class GIN_Conv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GIN_Conv, self).__init__(aggr = "add")
        self.in_dim = in_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        # torch.nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr != None:
            edge_embedding = self.edge_encoder(edge_attr)    
        else:
            edge_embedding = None
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x , edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr != None:
            return F.relu(x_j + edge_attr)
        else:
            return x_j    
    def update(self, aggr_out):
        return aggr_out


class Weight(nn.Module):
    def __init__(self, nums):
        super(Weight, self).__init__()
        self.nums = nums
        self.w = nn.Parameter(torch.ones(self.nums, 1, 1)/self.nums, requires_grad = True)
        self.lin_w = nn.Linear(self.nums, 1)
        self.lin = nn.Sequential(nn.Linear(32, 32),nn.ReLU(),nn.Linear(32,32))
    def forward(self, x):
        # x = self.lin(x)
        x = F.normalize(x)
        # x = F.dropout(x)
        # out = self.lin_w(x.permute(2,1,0)).permute(2,1,0)
        out = self.w * x
        return out.sum(0)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(MLP, self).__init__()
        self.lin = nn.Linear(in_dim, hid_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class GNN_Backbone(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers = 2, gnn = 'gcn'):
        super(GNN_Backbone, self).__init__()
        self.embedding_layer = nn.Embedding(30000, 64)
        self.lin_layer = nn.Linear(in_dim, hid_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = num_layers
        self.gnn = gnn
        for i in range(self.num_layers):
            if gnn == 'gcn':
                self.convs.append(GCNConv(hid_dim, hid_dim))
            elif gnn == 'gat':
                self.convs.append(GATConv(hid_dim, hid_dim))
            elif gnn == 'sage':
                self.convs.append(SAGEConv(hid_dim, hid_dim))
            elif gnn == 'gatv2':
                self.convs.append(GATv2Conv(hid_dim, hid_dim, edge_dim = 1))
            elif gnn == 'gin':
                self.convs.append(GIN_Conv(hid_dim, hid_dim))
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
        for layer in range(self.num_layers):
            if self.gnn != 'sage':
                h = self.convs[layer](h, edge_index, edge_attr)
            else:
                h = self.convs[layer](h, edge_index)
            # h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, 0.5)
            else:
                h = F.dropout(F.relu(h), 0.5)
        
        return h

    def decoder(self, z, edge_index):
        edge_scores = torch.sigmoid(z[edge_index[0]] * z[edge_index[1]]).mean(1)
        return edge_scores.squeeze()

class VGAEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, gnn = 'gcn', num_layers = 2):
        super(VGAEModel, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.encoder = nn.ModuleList()
        self.x_encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        for _ in range(4):
            if gnn == 'gcn':
                self.encoder.append(GCNConv(hid_dim, hid_dim))
            elif gnn == 'gin':
                self.encoder.append(GIN_Conv(hid_dim, hid_dim))
            elif gnn == 'gat':
                self.encoder.append(GATConv(hid_dim, hid_dim))
            elif gnn == 'sage':
                self.encoder.append(SAGEConv(hid_dim, hid_dim))
            elif gnn == 'gatv2':
                self.encoder.append(GATv2Conv(hid_dim, hid_dim))

        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.embed_layer = nn.Embedding(50000, hid_dim)
        # self.batch_norm = torch.nn.BatchNorm1d(hid_dim)
    
    def agg_edge(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        src, dst = edge_index
        new_x = x.clone()
        new_x[src] = F.relu(x[src] + edge_embedding)
        new_x[dst] = F.relu(x[dst] + edge_embedding)
        return new_x

    def forward(self, x, edge_index, edge_attr):
        if x.shape[1] == 1:
            x = self.embed_layer(x.squeeze().long())
        else:
            x = self.x_encoder(x)
        x = self.encoder[0](x, edge_index, edge_attr)
        x = F.relu(x)
        mean = self.encoder[1](x, edge_index, edge_attr)
        logstd = self.encoder[2](x, edge_index, edge_attr)
        gaussian_noise = torch.randn(x.size(0), self.hid_dim).to(x.device)
        z = mean + gaussian_noise * torch.exp(logstd)
        
        return z
    
    def decoder(self, z, edge_index):
        edge_scores = torch.sigmoid(z[edge_index[0]] * z[edge_index[1]]).mean(1)
        return edge_scores.squeeze()


