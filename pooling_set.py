# code adapted from https://github.com/mint-vu/backbone_vs_pooling/blob/main/poolings/attention.py

from curses import use_default_colors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_geometric.nn import MessagePassing

from equiv_set import MLP
from utils import interp1d, sparse_sort
import time
import math
from torch_geometric.typing import OptTensor

class GAP(nn.Module):
    '''
        DeepSets pooling MLP(sum_s x_s). The encoder will deal with the inner MLP
    '''
    def __init__(self, use_mlp=False, d_in=None, d_out=None, num_layers=2, hidden_layer_size=512, norm_type='None', args=None):
        super().__init__()
        self.use_mlp = use_mlp
        self.net =  MLP(d_in, hidden_layer_size, d_out, num_layers, dropout=args.dropout, Normalization=norm_type, InputNorm=False) if use_mlp else nn.Identity()

    def forward(self, x, hyperedge_index, data, name):
        # x: m x d, where m is the number of incidence connections 
        x = scatter_mean(x,hyperedge_index[1], dim=0)
        x = self.net(x)
        return x

    def reset_parameters(self):
        if self.use_mlp:
            self.net.reset_parameters()



class FPSWE_pool(nn.Module):
    def __init__(self, d_in,  num_anchors=1024, num_projections=1024, anch_freeze=True, out_type='linear'):
        '''
        The PSWE and LPSWE module that produces 
        fixed-dimensional permutation-invariant embeddings 
        for input sets of arbitrary size.
        '''

        super(FPSWE_pool, self).__init__()
        self.d_in = d_in # the dimensionality of the space that each set sample belongs to
        self.num_ref_points = num_anchors # number of points in the reference set
        self.num_projections = num_projections # number of slices
        self.anch_freeze = anch_freeze # if True the reference set and the theta are not learnable

        uniform_ref = torch.linspace(-1, 1, num_anchors).unsqueeze(1).repeat(1, num_projections) #num_anchors x num_preojections
        self.reference = nn.Parameter(uniform_ref, requires_grad=not self.anch_freeze)

        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        if num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_v.requires_grad = not self.anch_freeze

        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_anchors))
        nn.init.xavier_uniform_(self.weight)
        
        self.deg_helper = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        device = self.weight.device

        if self.anch_freeze == False:
            uniform_ref = torch.linspace(-1, 1, self.num_ref_points).unsqueeze(1).repeat(1, self.num_projections).to(device) #num_anchors x num_preojections
            self.reference.data = uniform_ref

        if self.num_projections <= self.d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        

    def double_self_loops(self, features, index):
        '''
        for the isolated nodes double them because pooling only works
            for minimum 2 elements in a set
        '''
        # Find indices where the group appears only once
        counts = torch.bincount(index)
        unique_mask = counts[index] == 1

        # Get elements to be duplicated
        unique_features = features[unique_mask]
        unique_groups = index[unique_mask]
        
        # Stack original tensor with duplicated unique elements
        new_features = torch.cat((features, unique_features), dim=0)
        new_index = torch.cat((index, unique_groups), dim=0)
        return new_features, new_index

    def forward(self, X, hyperedge_index, data, name):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  N x dn tensor containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: E x num_projections tensor, containing an embedding of dimension "num_projections" (i.e., number of slices)
        '''
   
        # Step 1: project samples into the 1D slices
        N, dn = X.shape
        Xslices = self.get_slice(X) # N x num_projections

        # for the self-loops doible the node to be able to apply the pooling 
        Xslices, hyperedge_index_new = self.double_self_loops(Xslices, hyperedge_index[1])
        Xslices_sorted, Xind = sparse_sort(Xslices, hyperedge_index_new)

        # regardless of the column sorting, all of them should have the same resorted index
        hyperedge_index_1_sorted = hyperedge_index_new[Xind[:,0]]
        M, dm = self.reference.shape

        eps = 0.00001
        #this should allow a correct interpolation when M>N
        margin_up = 0.9999
        assert (margin_up+eps < 1)

        # basically if i am at the first iteration precompute every constant i will use to be faster
        if self.deg_helper == None:
            self.deg_helper = torch.ones_like(hyperedge_index_1_sorted)
            self.R = torch.arange(hyperedge_index_1_sorted.shape[0]).to(X.device).to(torch.float64)+1
            self.pad = torch.tensor([0.0]).to(X.device)
            self.edges = torch.sort(torch.unique(hyperedge_index_1_sorted))[0]
            self.hyperedge_index_anchors_1 = self.edges.repeat_interleave(M)
            self.num_edges = self.edges.shape[0]
            self.xnew = torch.linspace(0, 1, M).repeat(self.num_edges).to(X.device).to(torch.float64)
            self.xnew = self.xnew * 0.99998+eps

            self.ynew = torch.zeros((self.num_projections, M*self.num_edges)).to(X.device)

            max_edge_index = self.edges.max()+1
            self.out1 =  torch.zeros((max_edge_index, self.num_projections)).to(X.device)
 
        num_edges = self.num_edges
        R = self.R

        # compute the degree
        D1 = scatter_add(self.deg_helper, hyperedge_index_1_sorted) #E
        D = torch.index_select(D1, 0,  hyperedge_index_1_sorted)

        # Step 2: interpolate 

        # compute the x indices to be used as positions for interpolation
        # they are computer for each hyperedge in parallel and are uniformly arranged
        ptr = torch.cat((self.pad,torch.cumsum(D1, dim=0)))
        P = torch.index_select(ptr, 0,  hyperedge_index_1_sorted)
        assert (D.min() >= 2)
        x = (R-P-1)/(D-1)*0.99999+eps +hyperedge_index_1_sorted
        x = x.unsqueeze(0).repeat(self.num_projections, 1)

        hyperedge_index_anchors_1 = self.hyperedge_index_anchors_1
        xnew = self.xnew + hyperedge_index_anchors_1
        xnew = xnew.unsqueeze(0).repeat(self.num_projections, 1)

        #this still correspond to hyperedge_index_1_sorted
        y = torch.transpose(Xslices_sorted, 0, 1).reshape(self.num_projections, -1)
        
        # interpolate y based on the x values
        Xslices_sorted_interpolated = interp1d(x, y, xnew,self.ynew,hyperedge_index_1_sorted).view(self.num_projections, -1)
        Xslices_sorted_interpolated = torch.transpose(Xslices_sorted_interpolated, 0, 1)

        # reshape the (projected) references. no need for projection since we sample them already projected
        Rslices = self.reference.unsqueeze(0).repeat(num_edges,1,1)#.to(X.device) # num_edges x M x num_projections
        Rslices = Rslices.reshape(num_edges*M,-1) # num_edges x  x num_projections

        # Step 3: sort the references and compute the distance
        _, Rind = sparse_sort(Rslices, hyperedge_index_anchors_1) #num_edges*M x num_projections
 
        # compute the distance between the sorted samples
        embeddings = Rslices - torch.gather(Xslices_sorted_interpolated, dim=0, index=Rind)
        embeddings = embeddings.transpose(0, 1) #num_projections x num_edges*M
        embeddings = embeddings.reshape(-1, num_edges, M) #num_projections x num_edges x M

        # Step 4: weighted sum of all samples
        w = self.weight.unsqueeze(1).repeat(1,num_edges,1)        
        weighted_embeddings = (w * embeddings) #num_projections x num_edges x M
        weighted_embeddings = weighted_embeddings.mean(-1) #num_projections x num_edges
        out = weighted_embeddings.transpose(0,1)

        final_out = self.out1.clone()
        final_out[self.edges,:]  = out
        return final_out
        

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)


class PMAPool(MessagePassing):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):

        super(PMAPool, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.aggr = 'add'

        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads*self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads*self.hidden)
        self.att_r = Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=self.dropout, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)

        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x,  edge_index, data, name):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.hidden
        # print(name)
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C) # M x H x C
            x_V = self.lin_V(x).view(-1, H, C) # M x H x C
            alpha_r = (x_K * self.att_r).sum(dim=-1)

        edge_index_0 = torch.arange(edge_index.shape[1]).to(x.device)
        edge_index_1 = edge_index[1]
        edge_index_new = torch.stack([edge_index_0, edge_index_1], dim=0)

        out = self.propagate(edge_index_new, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        self._alpha = None
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out)))

        return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j):

        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
#         ipdb.set_trace()
        if self.aggr is None:
            raise ValeuError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



