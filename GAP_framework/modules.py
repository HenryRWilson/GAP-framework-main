import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from layers import MeanAggregator, Encoder
from utils import featureless_random_graph,get_feature_func,get_neigh_list
"""
To begin with let's try to implement the GAP-scalefree model
which uses GraphSAGE layers.
Training it on undirected graphs first let's us simplify the model
so we will do the GAP-scalefree model on a E-R produced graph
"""
class GraphEmbeddingModule(nn.Module):
    """
    This part of the network takes in a graph G = (V,E) and 
    produces node embeddings of each node in the network using 
    classical GraphSAGE layers.
    We first implement a 4 layers of GraphSAGE with 128 hidden units.
    """
    def __init__(self,
                features,
                feature_dim,
                adj_lists,
                embed_dim=64,
                hidden_dim=128,
                num_sample=2,
                cuda=False):
        """
        Note that the features as aggregated by the last layer are passed to the next
        layer using a lambda function.
        This means when we call the final layer, it will recursively call the
        preceding layers.
        """
        # TODO : This can be neatened up a lot!
        super().__init__()
        self.agg1 = MeanAggregator(features,cuda=cuda)
        self.gSAGE1 = Encoder(features,
                              feature_dim,
                              hidden_dim,
                              adj_lists,
                              self.agg1,
                              num_sample=num_sample,
                              cuda=cuda)
        self.agg2 = MeanAggregator(lambda nodes : self.gSAGE1(nodes).t(),
                                   cuda=cuda,verbose=True)
        self.gSAGE2 = Encoder(lambda nodes : self.gSAGE1(nodes).t(),
                              hidden_dim,
                              hidden_dim,
                              adj_lists,
                              self.agg2,
                              num_sample=num_sample,
                              cuda=cuda,
                              base_model=self.gSAGE1,
                             verbose=True)
        self.agg3 = MeanAggregator(lambda nodes : self.gSAGE2(nodes).t(),cuda=cuda)
        self.gSAGE3 = Encoder(lambda nodes : self.gSAGE2(nodes).t(),
                              hidden_dim,
                              hidden_dim,
                              adj_lists,
                              self.agg3,
                              num_sample=num_sample,
                              cuda=cuda,
                             base_model=self.gSAGE2)
        self.agg4 = MeanAggregator(lambda nodes : self.gSAGE3(nodes).t(),cuda=cuda)
        self.gSAGE4 = Encoder(lambda nodes : self.gSAGE3(nodes).t(),
                              hidden_dim,
                              embed_dim,
                              adj_lists,
                              self.agg4,
                              num_sample=num_sample,
                              cuda=cuda,
                             base_model=self.gSAGE3)
        
    def forward(self,n):
       """
        Here we only need to call the last layer to call all!
       """
       embeddings = self.gSAGE4(n)
       return embeddings.t()

    def change_graph(self,new_adj,new_features):
       """
        Helper method to specialise to different graphs
       """
       self.gSAGE1.adj_lists = new_adj
       self.gSAGE2.adj_lists = new_adj
       self.gSAGE3.adj_lists = new_adj
       self.gSAGE4.adj_lists = new_adj

       self.gSAGE1.features = new_features
       self.agg1.features = new_features


class GraphPartitioningModule(nn.Module):
    """
    This part of the network takes in the node embeddings
    and produces probabilities of the node belonging to each
    state.
    Roughly a transform from
    [n_nodes, embedding_space_dim] --> [n_nodes,p_per_partition]
    """
    def __init__(self,latent_dim,n_partitions):
        super().__init__()
        self.dense = nn.Linear(latent_dim,n_partitions)
    
    def forward(self,embeddings):
        embeddings = self.dense(embeddings)
        return F.softmax(embeddings,dim=1)

                            
""" Unit tests! """
if __name__ == '__main__':
    adj, degrees, feats = featureless_random_graph(100)
    adj_list = get_neigh_list(adj)
    node_list = [*range(adj.shape[0])]
    input_nodes = [*range(adj.shape[0])]
    to_remove = []
    for i in input_nodes:
        if len(adj_list[i]) == 0:
            print(f"Disconnected Node {i}")
            to_remove.append(i)
    for i in to_remove[::-1]:
        input_nodes.pop(i)
    features = get_feature_func(node_list,feats)
    test_E = GraphEmbeddingModule(features,
                      feats.shape[1],
                      adj_list,
                      cuda=False)
    ## Let's embed a whole graph as a test
    x = test_E(input_nodes)

    # And feed this to a Graph Partition module
    n_categories = 5
    latent = 5
    test_G = GraphPartitioningModule(latent,n_categories)
    prob = test_G(x)
