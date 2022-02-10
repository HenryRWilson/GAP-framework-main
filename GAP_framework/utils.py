import torch
import os
import networkx as nx
from matplotlib import pyplot as plt
from networkx.generators import erdos_renyi_graph as er_graph
from networkx.generators import scale_free_graph as sfg
from networkx.linalg import adjacency_matrix
from sklearn.decomposition import PCA
from collections import Counter

"""
hello
Here we will: 
- Preprocess generated graphs
- Implement methods to measure results
"""
def featureless_random_graph(n_nodes,prob=0.1,
                kind="er",
                feature_size=10):
    """
    Generates a random graph and then performs pca
    on it and stores this as a feature matrix.
    Returns: 
      -->   adjacency matrix    [n_nodes,n_nodes]
      -->   node degree matrix  [n_nodes,1]
      -->   node feature matrix [n_nodes,feature_size] 
      in that order.
    Kind is either sf: Scale Free
                   er: Erdos-Renyi
    Features size :int: is the dimension of PCA to return.
    """
    # Generate graph
    if kind == "sf":
        G = sfg(n_nodes)
    elif kind == "er":
        G = er_graph(n_nodes, prob)
    else:
        print(f"Kind: {kind} not understood")
        raise Exception

    # Generate adjacency matrix
    adjacency_m = adjacency_matrix(G) # This returns a SPARSE matrix!
    adjacency_m = adjacency_m.toarray()
    adjacency_m = torch.tensor(adjacency_m)

    # Generate node degree matrix
    node_degrees = [*G.degree()]
    node_degrees = torch.tensor(node_degrees).float()

    # Generate node features
    pca = PCA(n_components=feature_size)
    node_features = pca.fit_transform(adjacency_m)
    node_features = torch.tensor(node_features)

    return adjacency_m, node_degrees, node_features

# Utils
def get_feature_func(nodes,features):
    """
    Helper function to allow the node to call initial features like it would a
    layer output.
    """
    def feature_func(node):
        return features[node]
    return feature_func

def get_neigh_list(adj_m,to_set=True):
    adj_list = []
    for node in adj_m.tolist():
        neigh_list = []
        for neigh_id in range(len(node)):
            if node[neigh_id] != 0:
                if not to_set:
                    for j in range(node[neigh_id]):
                        neigh_list.append(neigh_id)
                else:
                    neigh_list.append(neigh_id)
        adj_list.append(set(neigh_list))
    return adj_list

def image_graph(adjacency_matrix,run_name,i,predictions=None):
    G = nx.Graph(adjacency_matrix.detach().numpy())
    path = f"./graphs/{run_name}/"
    if predictions is not None:
        test_labels = gen_test_labels(predictions)
    else: 
        test_labels = None
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.figure(figsize=(12,12))
    nx.draw(G,node_color=test_labels,with_labels=True)
    plt.savefig(path+str(i)+"-result.png")
    plt.close()

def gen_graph(n_nodes,run_name,epoch,save=True):
    # Generate a random graph
    adj, degrees, feats = featureless_random_graph(n_nodes)
    adj_list = get_neigh_list(adj)
    node_list = [*range(adj.shape[0])]
    input_nodes = [*range(adj.shape[0])]
    features = get_feature_func(node_list,feats)
    if save and run_name:
        path = f"./graphs/{run_name}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.figure(figsize=(12,12))
        nx.draw(nx.Graph(adj.detach().numpy()),with_labels=True)
        plt.savefig(path + str(epoch) + ".png")
        plt.close()
    return input_nodes, adj_list, features, degrees, adj

def gen_test_labels(probabilities):
    test_labels = [torch.argmax(p).item() for p in probabilities]
    colour = []
    for p in test_labels:
        if p == 0:
            colour.append("blue")
        if p == 1:
            colour.append("red")
        if p == 2:
            colour.append("yellow")
        if p == 3:
            colour.append("olive")
        if p == 4:
            colour.append("cyan")
    return colour

def rate_preds(adj,preds):
    n_nodes = adj.shape[0]
    # Take the predicted state
    states = [max(p) for p in preds]
    # Get edge-list
    edges = adj.nonzero(as_tuple=False).numpy()
    # Calc edge cut
    edge_cut = calc_edge_cut(edges,states)
    # Calc balance
    balance = calc_balance(states)
    return edge_cut, balance

def calc_edge_cut(edges,states):
    """
    Edge cut is the ratio of the cut to the number of edges.
    edges should be a iterable of iterables of edge indices.
    states should be a list of the predicted states
    """
    n_edges = len(edges)
    states = {i:s for i,s in enumerate(states)}
    cuts = 0
    for (state1,state2) in edges:
        if states[state1] != states[state2]:
            cuts += 1
    return cuts / n_edges


def calc_balance(states):
    """
    From what I can gather:
    B = 1 - MSE(node_populations,n_nodes/n_parts)
    B = 1 - 1/n*sum(node_pop - n_nodes/n_parts)**2
    """
    n_nodes = len(states)
    node_pops = Counter(states)
    balance = n_nodes / len(node_pops.keys())
    mse = sum((v - balance)**2 for v in node_pops.values()) 
    mse *= 1 / n_nodes
    return 1 - mse

def process_matlab_graph(file_path, n_features = 100):
    pass

def matlab_graph(file_path,run_name,epoch,save=True):
    # Generate a random graph
    adj, degrees, feats = process_matlab_graph(file_path)
    adj_list = get_neigh_list(adj)
    node_list = [*range(adj.shape[0])]
    input_nodes = [*range(adj.shape[0])]
    features = get_feature_func(node_list,feats)
    if save and run_name:
        path = f"./graphs/{run_name}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.figure(figsize=(12,12))
        nx.draw(nx.Graph(adj.detach().numpy()),with_labels=True)
        plt.savefig(path + str(epoch) + ".png")
        plt.close()
    return input_nodes, adj_list, features, degrees, adj

if __name__ == '__main__':
    """
    Unit tests!
    """
    adj,degrees,features = featureless_random_graph(100)
    print("Adjacency: ", adj.shape)
    print("Degrees: " ,  degrees.shape)
    print("Features: ",  features.shape)

    # Getting into right format for layers
    adj_list = get_neigh_list(adj)
    node_list = [*range(adj.shape[0])]
    features = get_feature_func(node_list,features)

