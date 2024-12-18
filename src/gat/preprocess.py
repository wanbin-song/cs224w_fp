import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import networkx as nx
import pandas as pd
import random
import glob
import os

def generate_graph(file_path):
    facebook = pd.read_csv(
        file_path,
        sep=" ",
        names=["start_node", "end_node"],
    )
    G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")
    return G

def generate_subgraph(file_path):
    G = nx.read_edgelist(file_path, nodetype=int)
    return G

def generate_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)
    # return data, DataLoader([data], batch_size=64, shuffle = True)

def sample_negative_edges(G, num_neg_samples):
    neg_edge_list = set() 
    edge_set = set(G.edges)  
    node_list = list(G.nodes)
    while len(neg_edge_list) < num_neg_samples:
        u, v = random.sample(node_list, 2)  
        if (u, v) not in edge_set and (v, u) not in edge_set: 
            neg_edge_list.add((u, v))  

    return list(neg_edge_list)

