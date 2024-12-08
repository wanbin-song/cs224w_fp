import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import pandas as pd
import glob
import os
from preprocess import generate_graph, generate_data, sample_negative_edges
from sklearn.model_selection import train_test_split
from gat import GAT # GNNStack
from train import train
from evaluate import evaluate
import json

if __name__ == "__main__":
    G = generate_graph("/ds1/data_f/facebook_combined.txt")
    data = generate_data(G)
    train_edges, test_edges = train_test_split(np.array(G.edges), test_size=0.2, random_state=129)
    pos_train_edges = torch.tensor(train_edges, dtype = torch.long).t()
    neg_train_edges = torch.tensor(sample_negative_edges(G, len(train_edges)), dtype = torch.long).t()
    # args = {'num_layers':5, 'heads':2, 'dropout':0.5}
    # model = GNNStack(data.num_node_features, 32 , 2, args)
    model = GAT(data.num_node_features, 32, 2 , 2)
    # optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.1, weight_decay=0.005)
    lr = 0.01
    weight_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion =  nn.BCEWithLogitsLoss()
    model_path = "./best_gat.pth"
    train(G, data, pos_train_edges, neg_train_edges, model, optimizer, criterion, model_path, f"gat_train_log_{lr}_{weight_decay}.txt")
    pos_test_edges = torch.tensor(test_edges, dtype = torch.long).t()
    neg_test_edges = torch.tensor(sample_negative_edges(G, len(test_edges)), dtype = torch.long).t()
    model = GAT(data.num_node_features, 32, 2 , 2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    hit_rate, mrr = evaluate(G, data, pos_test_edges, neg_test_edges, model)
    with open(f"eval_{lr}_{weight_decay}", "w") as f:
        json.dump({"hit rate": hit_rate, "mmr": mrr}, f)