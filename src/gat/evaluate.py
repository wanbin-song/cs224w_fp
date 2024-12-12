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

def evaluate(G, data, pos_edges, neg_edges, model):
    test_edges = torch.cat([pos_edges, neg_edges], dim = 1)
    labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
    model.eval()
    out = model(data)
    preds = torch.sigmoid(torch.sum(out[test_edges[0]] * out[test_edges[1]], -1))

    pos_scores = (out[pos_edges[0]] * out[pos_edges[1]]).sum(dim=1)

    neg_scores = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    hit_rate = (sorted_labels[:100] == 1).float().mean().item()

    ranks = (sorted_labels == 1).nonzero(as_tuple=True)[0] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    print(f"Hit-Rate: {hit_rate:.4f}, MRR: {mrr:.4f}")

    return hit_rate, mrr
