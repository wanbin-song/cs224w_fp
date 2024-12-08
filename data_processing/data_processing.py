import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import glob
import os

def get_summary(G):
    summary = ""
    summary += f"Number of Nodes : {G.number_of_edges()}\n"
    summary += f"Number of Edges : {G.number_of_nodes()}\n"
    summary += f"Number of Connected Components : {nx.number_connected_components(G)}\n"
    summary += f"Size of the Largest Connected Compopnent : {max([ len(i) for i in list(nx.connected_components(G))])}"
    return summary

edge_file_list = [f for f in glob.glob("/ds1/data_f/facebook/*.edges")]

for edge_file in edge_file_list:
    edge = int(os.path.basename(edge_file).split('.')[0])
    G = nx.read_edgelist(edge_file, nodetype=int)
    nx.draw(G, node_color = "lightblue", with_labels = True, node_size = 200)
    fig = plt.figure(1, figsize=(500, 250))
    summary = get_summary(G)
    print(summary)
    fig.text(x = 0, y = 0, s = summary)
    fig.savefig(f"{edge}_graph.png")