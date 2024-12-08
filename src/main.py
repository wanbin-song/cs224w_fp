import os
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

def load_graph(file_path):
    with gzip.open(file_path, 'rt') as f:
        return nx.read_edgelist(f, create_using=nx.Graph(), nodetype=int)

def create_pyg_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def negative_sampling(G, all_nodes, num_samples):
    negative_samples = []
    while len(negative_samples) < num_samples:
        u, v = np.random.choice(all_nodes, 2, replace=False)
        if not G.has_edge(u, v):
            negative_samples.append([u, v])
    return negative_samples

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x
    
def initialize_weights(m):
    if isinstance(m, (nn.Linear, GCNConv)):
        nn.init.xavier_uniform_(m.weight)

def train_gnn(model, data, train_edges, G, all_nodes, num_epochs=100, lr=0.01, output_dir="experiments/gcn_experiment/lre2"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(output_dir, exist_ok=True)
    training_log_path = os.path.join(output_dir, "training_logs.txt")

    with open(training_log_path, "w") as log_file:
        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            model.train()
            optimizer.zero_grad()

            pos_edges = torch.tensor(train_edges, dtype=torch.long).t()
            pos_scores = torch.sigmoid((model(data)[pos_edges[0]] * model(data)[pos_edges[1]]).sum(dim=1))

            neg_edges = torch.tensor(negative_sampling(G, all_nodes, len(train_edges)), dtype=torch.long).t()
            neg_scores = torch.sigmoid((model(data)[neg_edges[0]] * model(data)[neg_edges[1]]).sum(dim=1))

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                log_file.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}\n")
                log_file.flush()

def evaluate(model, data, test_edges, negative_edges, output_dir="experiments/gcn_experiment"):
    model.eval()

    pos_edges = torch.tensor(test_edges, dtype=torch.long).t()
    pos_scores = (model(data)[pos_edges[0]] * model(data)[pos_edges[1]]).sum(dim=1)

    neg_edges = torch.tensor(negative_edges, dtype=torch.long).t()
    neg_scores = (model(data)[neg_edges[0]] * model(data)[neg_edges[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    hit_rate = (sorted_labels[:len(pos_scores)] == 1).float().mean().item()

    ranks = (sorted_labels == 1).nonzero(as_tuple=True)[0] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    print(f"Hit-Rate: {hit_rate:.4f}, MRR: {mrr:.4f}")

    results = {
        "Hit-Rate": hit_rate,
        "MRR": mrr
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

file_path = './data/facebook_combined.txt.gz'
G = load_graph(file_path)
data = create_pyg_data(G)

all_nodes = list(G.nodes)
edges = np.array(G.edges)
train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)
negative_edges = np.array(negative_sampling(G, all_nodes, len(test_edges)))

model = GCN(input_dim=data.x.size(1), hidden_dim=16, output_dim=16)
train_gnn(model, data, train_edges, G, all_nodes)

evaluate(model, data, test_edges, negative_edges)
