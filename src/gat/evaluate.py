import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.nn as pyg_nn
from tqdm import tqdm

def calculate_hit_rate(preds, labels):
    preds = (preds.detach().numpy() > 0.5).astype(int)
    labels = labels.detach().numpy()
    true_p = ((labels == 1) & (preds == 1)).sum()
    total_p = (labels == 1).sum()
    return true_p / total_p if total_p > 0 else 0

def calculate_mrr(preds, labels):
    preds = preds.detach().numpy()
    labels = labels.detach().numpy()
    ranks = []
    for pred, label in zip(preds, labels):
        if label == 1:
            ranks.append(1)
        else:
            ranks.append(0)
    return sum(ranks) / len(ranks) if len(ranks) > 0 else 0

def evaluate(G, data, pos_edges, neg_edges, model):
    test_edges = torch.cat([pos_edges, neg_edges], dim = 1)
    labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
    model.eval()
    out = model(data.x, data.edge_index)
    preds = torch.sigmoid(torch.sum(out[test_edges[0]] * out[test_edges[1]], -1))

    
    # hit_rate = calculate_hit_rate(preds, labels)
    # mrr = calculate_mrr(preds, labels)

    pos_scores = (model(data.x, data.edge_index)[pos_edges[0]] * model(data.x, data.edge_index)[pos_edges[1]]).sum(dim=1)

    neg_scores = (model(data.x, data.edge_index)[neg_edges[0]] * model(data.x, data.edge_index)[neg_edges[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    hit_rate = (sorted_labels[:len(pos_scores)] == 1).float().mean().item()

    ranks = (sorted_labels == 1).nonzero(as_tuple=True)[0] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    print(f"Hit-Rate: {hit_rate:.4f}, MRR: {mrr:.4f}")

    return hit_rate, mrr
