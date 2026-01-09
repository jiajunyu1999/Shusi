import torch.nn.functional as F
import torch
import numpy as np
import pickle
import pandas as pd
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import copy

def scale(z, dim):
    zmax = z.max(dim=dim, keepdim=True)[0]
    zmin = z.min(dim=dim, keepdim=True)[0]
    z = (z - zmin) / (zmax - zmin)
    return z

def get_data(cell, args, feat_map1, feat_map2, gene_map):
    # gene --> id int
    # gene_map = pickle.load(open('./processed_data/mapping_data/gene_mapping_all.pkl', 'rb'))
    # id int -> gene
    reversed_gene_map = {v: k for k, v in gene_map.items()}
    
    # cellline --> array vector
    # cellline_map = pickle.load(open('./processed_data/mapping_data/cellname_cancer.pkl','rb'))

    # gene --> embedding tensor
    if args.feat == 'g-anno':
        feat_map = pickle.load(open('./data/feat_data/embedding_anno_gene.pkl','rb'))
    elif args.feat == 'g-seq':
        feat_map = pickle.load(open('./data/feat_data/embedding_seq_galactica.pkl','rb'))        
    elif args.feat == 'd-seq-cls':
        feat_map = pickle.load(open('./data/feat_data/embedding_seq_bert_cls.pkl','rb'))        
    elif args.feat == 's-anno':
        # feat_map = pickle.load(open('./processed_data/feat_data/embedding_anno_sentence.pkl','rb'))        
        feat_map = copy.deepcopy(feat_map2)
    

    if args.feat == 'all':
        lens = 4096 + 768
    elif args.feat in ['g-anno', 's-anno', 'd-seq-cls', 'g-seq']:
        lens = feat_map[list(feat_map.keys())[0]].shape[0]

    gene_id = torch.LongTensor(np.load(cell+'_x.npy'))
    if args.feat == 'id':
        x = gene_id.unsqueeze(1)
    elif args.feat in ['g-anno', 'g-seq', 'd-seq-cls', 's-anno']:
        x = []
        for i in gene_id.numpy():
            if reversed_gene_map[i] in feat_map:
                x.append(feat_map[reversed_gene_map[i]].cpu().detach())
            else:
                x.append(torch.ones(lens))
        x = torch.stack(x, dim = 0).squeeze()
        x = scale(x, dim = 0)
        # x = F.normalize(x)
    elif args.feat == 'all':
        x = []       

        for i in gene_id.numpy():
            temp = []
            if reversed_gene_map[i] in feat_map1:
                temp.append(feat_map1[reversed_gene_map[i]].cpu().detach().squeeze())
            else:
                temp.append(torch.ones(4096))

            if reversed_gene_map[i] in feat_map2:
                temp.append(feat_map2[reversed_gene_map[i]].cpu().detach().squeeze())
            else:
                temp.append(torch.ones(768))
            temp = torch.cat(temp, dim = 0)
            x.append(temp)
        x = torch.stack(x, dim = 0).squeeze()
        x = scale(x, dim = 0)
        # x = F.normalize(x, dim = 1)
    edge_index = sp.load_npz(cell+'_edge.npz')
    edge_attr = torch.FloatTensor(edge_index.data)

    edge_attr = (edge_attr-edge_attr.min()) / (edge_attr.max() - edge_attr.min())
    edge_attr = edge_attr.unsqueeze(1)
    # edge_attr = F.normalize(edge_attr,dim=0)
    # edge_attr = bin_and_one_hot(edge_attr.squeeze().numpy(), 1000)

    edge_index = torch.tensor([edge_index.row, edge_index.col], dtype=torch.long)
    data = Data(x = x, id = gene_id, edge_index = edge_index, edge_attr=edge_attr)
    
    return data

def create_one_hot_encoding(tensor, num_bins):
    # Calculate bin edges
    bin_edges = np.linspace(np.min(tensor), np.max(tensor), num_bins + 1)

    # Digitize tensor into bins
    digitized = np.digitize(tensor, bin_edges[:-1])

    # Initialize an empty array for one-hot encoding
    one_hot_encoding = np.zeros((len(tensor), num_bins), dtype=int)

    # Assign one-hot vectors to the corresponding bins
    one_hot_encoding[np.arange(len(tensor)), digitized - 1] = 1

    return one_hot_encoding

def bin_and_one_hot(tensor, num_bins):
    bin_edges = np.linspace(np.min(tensor)-0.0001, np.max(tensor), num_bins + 1)
    bin_indices = np.digitize(tensor, bin_edges, right=True)
    one_hot_matrix = np.eye(num_bins)[bin_indices-1 ]
    return torch.FloatTensor(one_hot_matrix)

# def convert_to_directed_edge(edge_index, edge_attr=None):
#     # convert the undirected symmetric graph to the directed unsymmetric graph
#     '''
#         input: edge_index(tensor, [2, |E|]), edge_attr(tensor, [|E|])
#         output: directed_edge_index(tensor, [2, |E|/2]), directed_edge_attr(tensor, [|E|/2])
#     '''
#     unique_directed_edges, unique_indices = torch.sort(edge_index, dim=0)
#     unique_directed_edges, unique_indices = torch.unique(unique_directed_edges, dim=1, return_inverse=True)
#     unique_indices = unique_indices.cpu().detach().numpy()
#     unique_elements, first_occurrences = np.unique(unique_indices, return_index=True)
    
#     if edge_attr is not None:
#         directed_edge_attr = edge_attr[first_occurrences]
#     else:
#         directed_edge_attr = None
    
#     return unique_directed_edges, directed_edge_attr


def convert_to_directed_edge(edge_index, edge_attr=None):
    """
    更快地将无向对称图变成只保留唯一无向边形式的图。大量数据时速度也快。
    Args:
        edge_index: torch.LongTensor, shape [2, num_edges]
        edge_attr: Optional[Tensor], shape [num_edges]
    Returns:
        directed_edge_index: torch.LongTensor, shape [2, num_unique_edges]
        directed_edge_attr: Optional[Tensor], shape [num_unique_edges]
    """
    ei = edge_index
    min_e = torch.minimum(ei[0], ei[1])
    max_e = torch.maximum(ei[0], ei[1])

    # Create a unique key per undirected edge using a perfect hash
    # Note: use .item() to avoid dtype promotion surprises when adding 1
    max_node = ei.max().item()
    key = min_e.to(torch.int64) * (max_node + 1) + max_e

    # Get indices of the first occurrence of each unique key without using unsupported kwargs
    order = torch.argsort(key)
    sorted_key = key[order]
    unique_mask = torch.ones_like(sorted_key, dtype=torch.bool)
    unique_mask[1:] = sorted_key[1:] != sorted_key[:-1]
    unique_idx = order[unique_mask]

    unique_edges = torch.stack([min_e[unique_idx], max_e[unique_idx]], dim=0)
    directed_edge_attr = edge_attr[unique_idx] if edge_attr is not None else None
    return unique_edges, directed_edge_attr

def find_top_similar_nodes(embeddings, edge_index, top_k=100000):
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    similarity_matrix = similarity_matrix.triu(diagonal=1)  # 只保留上三角矩阵
    node_nums = embeddings.shape[0]
    
    exist_edges_set = set(map(tuple, edge_index.T.tolist()))
    
    score, indices = torch.topk(similarity_matrix.view(-1), k=top_k*2)
    row_indices = indices // node_nums
    col_indices = indices % node_nums
    unique_indices = torch.stack([row_indices, col_indices], dim=1)

    similar_edges_set = set(map(tuple, unique_indices.tolist()))
    diff_edge_set = torch.tensor(list(similar_edges_set - exist_edges_set))
    print(len(similar_edges_set), len(exist_edges_set), len(similar_edges_set&exist_edges_set), len(diff_edge_set))
    
    return diff_edge_set[0:top_k].cpu().detach().numpy(), score.cpu().detach().numpy(), len(similar_edges_set&exist_edges_set)/len(similar_edges_set)


def get_new_edge(node_embedding, model, cell_line):
    gene_map = pickle.load(open('/data7/chenyihan/katrina/processed_data/gene_mapping_all.pkl', 'rb'))
    reversed_gene_map = {v: k for k, v in gene_map.items()}
    proteins_map = pickle.load(open('/data7/chenyihan/katrina/processed_data/proteins_gene_map.pkl','rb'))
    reversed_proteins_map = {v: k for k, v in proteins_map.items()}

    ppi_edges = pd.read_csv('/data7/chenyihan/katrina/processed_data/ppi_edges.csv',header=None)
    ppi_edges = torch.LongTensor(np.array(ppi_edges)).T

    gene_edges = pd.read_csv('/data7/chenyihan/katrina/processed_data/exist_cell_line_edges.csv',header=None)
    gene_edges = torch.LongTensor(np.array(gene_edges)).T

    concat_edges = torch.concat([ppi_edges, gene_edges], dim=1)
    if not os.path.exists('/data7/chenyihan/katrina/new_edge/{}'.format(model)):
        os.mkdir('/data7/chenyihan/katrina/new_edge/{}'.format(model))
    with open(f'/data7/chenyihan/katrina/new_edge/{model}/{cell_line}.csv','w')as f:
        f.write('gene1,gene2,protein1,protein2,score\n')
    with open(f'/data7/chenyihan/katrina/new_edge/{model}/{cell_line}.csv','a+')as f:
        embeddings = torch.FloatTensor(np.array(node_embedding))    
        new_edge, score, recall = find_top_similar_nodes(embeddings, concat_edges)
        score /= score.max()
        for i,e in enumerate(new_edge):
            gene_name1, gene_name2 = reversed_gene_map[e[0]], reversed_gene_map[e[1]]
            try:
                proteins_name1 = reversed_proteins_map[gene_name1]
            except:
                proteins_name1 = ''
            try:
                proteins_name2 = reversed_proteins_map[gene_name2]
            except:
                proteins_name2 = ''
            f.write('{},{},{},{},{:.5f}\n'.format(gene_name1, gene_name2, proteins_name1, proteins_name2,score[i] ))
    return recall