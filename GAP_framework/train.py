import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import os
from tqdm import tqdm
from modules import GraphEmbeddingModule, GraphPartitioningModule
from utils import *
from datetime import datetime
class GAPmodel(nn.Module):
    """
    The full model.
    Simply links the two modules together
    """
    def __init__(self,
                features,
                feature_dim,
                adj_lists,
                n_partitions=2,
                embed_dim=64,
                hidden_dim=128,
                num_sample=10,
                cuda=True):
        super().__init__()
        self.embedding = GraphEmbeddingModule(
                            features,
                            feature_dim,
                            adj_lists,
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            num_sample=num_sample,
                            cuda=cuda
                        )
        self.partitioning = GraphPartitioningModule(
                            embed_dim,
                            n_partitions
                        )
    def forward(self,nodes):
        nodes = self.embedding(nodes)
        probability = self.partitioning(nodes)
        return probability

def kemeny_loss(): #determine variables
    pass

def expected_cut_loss(probabilities,
                      degree,
                      adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor([[*a] for a in adj_matrix])
    gamma = probabilities.t() @ degree
    yt = (1-probabilities).t()
    loss = (probabilities.div(gamma.t())) 
    loss = loss @ yt
    loss = torch.sum(loss * adj_matrix)
    return loss

if __name__ == '__main__':
    date = datetime.now()
    run_name = date.strftime("%b_%d_%H_%M") + "_run"
    # Training variables
    # TODO : Move into argparser
    n_graphs = 1   # n_unique graphs
    n_nodes = 100 # n_nodes in graph
    n_times = 100   # n_times exposed to each graph
    epoch_gen = tqdm(range(n_graphs))
    # Gen the first graph
    input_nodes, adj_list, features, degree, adj = gen_graph(n_nodes,run_name,0)
    features_inp = features(0).shape[0]
    # Init the model
    model = GAPmodel(features,
                      features_inp,
                      adj_list,
                      cuda=False)
    # Assign the optimizer
    optimizer = optim.Adam(model.parameters(),lr=7.5e-6)
    # Begin Training
    model.train()
    for i in epoch_gen:
        if i != 0:
            # Generate a new random graph
            input_nodes, adj_list, features, degree, adj = gen_graph(n_nodes,
                                                                     run_name,i)
            # Assign the model new features and adjacency
            model.embedding.change_graph(adj_list,features)
        for k in range(n_times):
            optimizer.zero_grad()
            node_probs = model(input_nodes)
            loss = expected_cut_loss(node_probs,degree[:,1],adj)
            loss.backward()
            optimizer.step()
            print("LOSS FOR THIS: ",loss.item())
    with torch.no_grad():
        image_graph(adj,run_name,i,predictions=node_probs)
        ec, balance = rate_preds(adj,node_probs)
        print("EDGE CUT :",ec,"BALANCE :",balance)
        with open("./graphs/scores.txt","a") as f:
            f.write(f"{run_name} {n_graphs} {n_nodes} {n_times} {ec} {balance}\n")

            

            
            



