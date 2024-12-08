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


def accuracy(pred, label):
  accu = 0.0
  accu = round((torch.sum((pred > 0.5).int() == label) / pred.shape[0]).item(), 4)
  return accu

def train(G, data, pos_edges, neg_edges, model, optimizer, criterion, model_path, log_path, num_epochs=100):
    train_edges = torch.cat([pos_edges, neg_edges], dim = 1)
    labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
    acc = None
    best_acc = 0.0
    model.train()
    with open(log_path, "w") as f:
        for epoch in tqdm(range(num_epochs), desc="Training GAT"):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            preds = torch.sigmoid(torch.sum(out[train_edges[0]] * out[train_edges[1]], -1))
            loss = criterion(preds, labels)
            acc = accuracy(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_result = f"Epoch: {epoch+1} | Loss: {loss.item()} | Accuracy: {acc}"
            f.write(epoch_result + "\n")
            f.flush() 
            print(epoch_result)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model_path)
        best_result = f"Model saved accuracy: {best_acc}"
        f.write(best_result+ "\n")
        f.flush() 
        print(best_result)
        