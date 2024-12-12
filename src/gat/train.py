import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
# def train(G, data, pos_edges, neg_edges, model, optimizer, criterion,  log_path, num_epochs=100):
#     train_edges = torch.cat([pos_edges, neg_edges], dim = 1)
#     labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
#     acc = None
#     best_acc = 0.0
#     best_model = model
#     model.train()
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
#     with open(log_path, "w") as f:
#         for epoch in tqdm(range(num_epochs), desc="Training GAT"):
#             optimizer.zero_grad()
#             out = model(data)
#             preds = torch.sigmoid(torch.sum(out[train_edges[0]] * out[train_edges[1]], -1))
#             loss = criterion(preds, labels)
#             acc = accuracy(preds, labels)
#             loss.backward()
#             optimizer.step()
#             # scheduler.step()
#             if (epoch+ 1)%10 == 0:
#               epoch_result = f"Epoch: {epoch+1} | Loss: {loss.item()} | Accuracy: {acc}"
#               f.write(epoch_result + "\n")
#               f.flush() 
#               # print(epoch_result)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_model = model
#                 # torch.save(model.state_dict(), model_path)
#         best_result = f"Model saved accuracy: {best_acc}"
#         f.write(best_result+ "\n")
#         f.flush() 
#         print(best_result)
#     return best_model

def train(G, data, pos_edges, neg_edges, model, optimizer, criterion, log_path, num_epochs=100):
    
    anchors = pos_edges[0]  
    positives = pos_edges[1]  
    negatives = neg_edges[0] 

    model.train()
    best_acc = 0.0
    best_model = model
    margin = 5.0
    criterion = ContrastiveLoss(margin = margin)
    with open(log_path, "w") as f:
        for epoch in tqdm(range(num_epochs), desc="Training"):
            optimizer.zero_grad()
            out = model(data)

            anchor_out = out[anchors]
            positive_out = out[positives]
            negative_out = out[negatives]

            pos_labels = torch.ones(anchor_out.size(0))  
            neg_labels = torch.zeros(negative_out.size(0))  

            loss_pos = criterion(anchor_out, positive_out, pos_labels)  
            loss_neg = criterion(anchor_out, negative_out, neg_labels)  
            loss = loss_pos + loss_neg 

            loss.backward()
            optimizer.step()

            pos_preds = nn.functional.pairwise_distance(anchor_out, positive_out)  
            neg_preds = nn.functional.pairwise_distance(anchor_out, negative_out)  

            pos_correct = (pos_preds < margin).float().sum()  
            neg_correct = (neg_preds >= margin).float().sum()  
            
            total_correct = pos_correct + neg_correct
            total_samples = anchor_out.size(0) + negative_out.size(0)
            acc = total_correct / total_samples 

            if (epoch + 1) % 10 == 0:
                epoch_result = f"Epoch: {epoch + 1} | Loss: {loss.item()} | Accuracy: {acc.item()}"
                f.write(epoch_result + "\n")
                f.flush()

            if acc > best_acc:
                best_acc = acc
                best_model = model
                # torch.save(model.state_dict(), model_path)

        best_result = f"Model saved accuracy: {best_acc.item()}"
        f.write(best_result + "\n")
        f.flush()
        print(best_result)

    return best_model

