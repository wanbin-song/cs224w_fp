import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import pandas as pd
import glob
import os
from preprocess import generate_graph, generate_subgraph, generate_data, sample_negative_edges#, create_dataloader
from sklearn.model_selection import train_test_split
from gat import GAT
from train import train
from evaluate import evaluate
import json
def main_graph():
    G = generate_graph("/ds1/data_f/facebook_combined.txt")
    data = generate_data(G)
    train_edges, test_edges = train_test_split(np.array(G.edges), test_size=0.2, random_state=42)
    pos_train_edges = torch.tensor(train_edges, dtype = torch.long).t()
    neg_train_edges = torch.tensor(sample_negative_edges(G, len(train_edges)), dtype = torch.long).t()
    dropout = 0.5
    num_layers = 5
    heads = 4
    model = GAT(data.num_node_features, 32, 2 , heads, num_layers = num_layers,dropout = dropout)
    lr = 0.001
    # weight_decay = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion =  nn.BCEWithLogitsLoss()
    num_epochs = 500
    model = train(G, data, pos_train_edges, neg_train_edges, model, optimizer, criterion, f"gat_train_log_lr{lr}_heads{heads}.txt", num_epochs = num_epochs)
    pos_test_edges = torch.tensor(test_edges, dtype = torch.long).t()
    neg_test_edges = torch.tensor(sample_negative_edges(G, len(test_edges)), dtype = torch.long).t()
    hit_rate, mrr = evaluate(G, data, pos_test_edges, neg_test_edges, model)
    with open(f"eval_lr{lr}_heads{heads}.json", "w") as f:
        json.dump({"hit rate": hit_rate, "mmr": mrr}, f)
def main_subgraph(G, data, node):
    train_edges, test_edges = train_test_split(np.array(G.edges), test_size=0.2, random_state=129)
    pos_train_edges = torch.tensor(train_edges, dtype = torch.long).t()
    neg_train_edges = torch.tensor(sample_negative_edges(G, len(train_edges)), dtype = torch.long).t()
    dropout = 0.7
    num_layers = 3
    model = GAT(data.num_node_features, 32, 2 , 2, num_layers = num_layers,dropout = dropout)
    lr = 0.001
    # weight_decay = 5e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion =  nn.BCEWithLogitsLoss()
    num_epochs = 500
    model = train(G, data, pos_train_edges, neg_train_edges, model, optimizer, criterion, f"gat_node{node}_train_log_lr{lr}.txt", num_epochs = num_epochs)
    pos_test_edges = torch.tensor(test_edges, dtype = torch.long).t()
    neg_test_edges = torch.tensor(sample_negative_edges(G, len(test_edges)), dtype = torch.long).t()
    hit_rate, mrr = evaluate(G, data, pos_test_edges, neg_test_edges, model)
    with open(f"eval_node{node}_lr{lr}.json", "w") as f:
        json.dump({"hit rate": hit_rate, "mmr": mrr}, f)
def to_networkx(data):
    G = nx.Graph()
    
    for i in range(data.num_nodes):
        G.add_node(i, feature=data.x[i].numpy())
        
    edge_list = data.edge_index.t().tolist()  
    G.add_edges_from(edge_list)
    
    return G
if __name__ == "__main__":
    # total_G = generate_graph("/ds1/data_f/facebook_combined.txt")
    # total_data = generate_data(total_G)
    # graphs = []
    # for node in [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]:
    #     G = generate_subgraph(f"/ds1/data_f/facebook/{node}.edges")
    #     graphs.append(G)
    # dataset = pad_and_adjust_subgraph(graphs,  total_data.x.size(0))
    # for i, data in enumerate(dataset):
    #     main_graph(graphs[i], data, node)
    
    main_graph()

