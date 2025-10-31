import numpy as np

from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import *

def edge_list_2_index(edge_list):
    """
    Args
    edge_list : a list of lists each representing nodes forming hyperedge
    returns :
    hyper_edge_index : tensor containing indices of incidence matrix [2, k]
    where k is number of non-zero values in incidence matrix
    """
    index = 0
    hyper_vertices_list = []
    hyper_edges_list = []

    num_nodes = max([max(x) for x in edge_list])+1

    for i in edge_list:
        for j in range(len(i)):
            hyper_edges_list += [index]
        hyper_vertices_list += i
        index+=1
    hyper_edge_index = torch.tensor([hyper_vertices_list, hyper_edges_list])
    return hyper_edge_index, num_nodes, len(edge_list)

def edge_list_2_VE(hgraph):
    node_list = []
    edge_list = []
    num_nodes = max([max(x) for x in hgraph])+1
    edge_idx = num_nodes

    for cur_he in hgraph:
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = int)
    return edge_index, num_nodes, len(hgraph)
        
class HyperEdgeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'extended_index':
            num_edges = self.num_hyperedges
            #be careful. value[0].max().item() + 1 might return smt else because some nodes are isolated
            num_nodes =  self.num_nodes
            num_incidence = self.edge_index.shape[1]
            return torch.tensor([[num_nodes], [num_nodes], [num_edges], [num_incidence]])
        if key == 'reversed_extended_index':
            num_edges = self.num_hyperedges
            #be careful. value[0].max().item() + 1 might return smt else because some nodes are isolated
            num_nodes =  self.num_nodes
            num_incidence = self.edge_index.shape[1]
            return torch.tensor([[num_edges], [num_edges], [num_nodes], [num_incidence]])
        elif key == 'edge_index':
            num_edges = self.num_hyperedges
            #be careful. value[0].max().item() + 1 might return smt else because some nodes are isolated
            num_nodes =  self.num_nodes
            return torch.tensor([[num_nodes], [num_edges]])
        else:
            return super(HyperEdgeData, self).__inc__(key, value)

