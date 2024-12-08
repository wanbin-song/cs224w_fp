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

# Step 1: Load Data
file_path = './data/facebook_combined.txt.gz'

with gzip.open(file_path, 'rt') as f:
    G = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=int)

# Step 2: Prepare Graph Data
# Convert NetworkX graph to PyTorch Geometric Data object
def create_pyg_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)  # Identity matrix for initial node features
    data = Data(x=x, edge_index=edge_index)
    return data

data = create_pyg_data(G)

# Step 3: Define GNN Model with Novelty
class EnhancedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedGCN, self).__init__()
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

# Step 4: Train-Test Split and Sampling for Recommendations
edges = np.array(G.edges)
all_nodes = list(G.nodes)
train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)

# Negative sampling for evaluation
def negative_sampling(G, num_samples):
    negative_samples = []
    while len(negative_samples) < num_samples:
        u, v = np.random.choice(all_nodes, 2, replace=False)
        if not G.has_edge(u, v):
            negative_samples.append([u, v])
    return negative_samples

negative_edges = np.array(negative_sampling(G, len(test_edges)))

# Step 5: Training
from tqdm import tqdm

def train_gnn(model, data, train_edges, num_epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()

        # Positive edges
        pos_edges = torch.tensor(train_edges, dtype=torch.long).t()
        pos_scores = torch.sigmoid((model(data)[pos_edges[0]] * model(data)[pos_edges[1]]).sum(dim=1))

        # Negative edges
        neg_edges = torch.tensor(negative_sampling(G, len(train_edges)), dtype=torch.long).t()
        neg_scores = torch.sigmoid((model(data)[neg_edges[0]] * model(data)[neg_edges[1]]).sum(dim=1))

        # Combine scores
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

model = EnhancedGCN(input_dim=data.x.size(1), hidden_dim=16, output_dim=16)
train_gnn(model, data, train_edges)

# Step 6: Evaluation
def evaluate(model, data, test_edges, negative_edges):
    model.eval()

    # Positive scores
    pos_edges = torch.tensor(test_edges, dtype=torch.long).t()
    pos_scores = (model(data)[pos_edges[0]] * model(data)[pos_edges[1]]).sum(dim=1)

    # Negative scores
    neg_edges = torch.tensor(negative_edges, dtype=torch.long).t()
    neg_scores = (model(data)[neg_edges[0]] * model(data)[neg_edges[1]]).sum(dim=1)

    # Combine
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])

    # Calculate Hit-Rate and MRR
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    hit_rate = (sorted_labels[:len(pos_scores)] == 1).float().mean().item()

    ranks = (sorted_labels == 1).nonzero(as_tuple=True)[0] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    print(f"Hit-Rate: {hit_rate:.4f}, MRR: {mrr:.4f}")

evaluate(model, data, test_edges, negative_edges)
