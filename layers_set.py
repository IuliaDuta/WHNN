import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling_set import *
from equiv_set import *

import pdb
        


class HyperLayer(nn.Module):
    """
    One layer of WHNN using 
        proc_type_V2E and proc_type_E2V as encoder
        pooling_type_V2E and pooling_type_E2V as pooling
    """

    def __init__(self, proc_type_V2E, pooling_type_V2E, proc_type_E2V, pooling_type_E2V, args):
        super(HyperLayer, self).__init__()
        self.alpha = args.restart_alpha # the coefficient for the residual connection
        self.dropout = args.dropout # dropout rate
        self.mlp3_layers = args.MLP3_num_layers # number of layers for the final projection (the update in the mpnn framework)

        self.normalization = args.normalization # the type of normalisation
        input_norm = args.deepset_input_norm

        # node-to-edge stage using proc_type_V2E encoder and pooling_type_V2E aggregator
        self.V2EConvs = SetLayer(d_in=args.MLP_hidden, 
                                        d_out = args.MLP_hidden, 
                                        num_layers=args.MLP_num_layers, 
                                        d_hid = args.MLP_hidden,
                                        proc_type = proc_type_V2E,
                                        pooling_type = pooling_type_V2E,
                                        args=args)

        # edge-to-node stage using proc_type_E2V encoder and pooling_type_E2V aggregator
        self.E2VConvs = SetLayer(d_in=args.MLP_hidden, 
                                        d_out = args.MLP_hidden, 
                                        num_layers=args.MLP_num_layers, 
                                        d_hid = args.MLP_hidden,
                                        proc_type = proc_type_E2V,
                                        pooling_type = pooling_type_E2V,
                                        args=args)
        # the update function in the end
        if self.mlp3_layers > 0:
            self.W = MLP(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden, self.mlp3_layers,
                dropout=self.dropout, Normalization=self.normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()


    def forward(self, x, x0, edge_index, reversed_edge_index, data):
        # node-to-edge
        h = self.V2EConvs(x, None, edge_index, data, 'V2E')
        # edge-to-node
        x = self.E2VConvs(h, x, reversed_edge_index, data, 'E2V')
       

        # this is in case some nodes at the end are isolated and they get dropped from the scatter output
        if x.shape[0] < x0.shape[0]:
            dif = x0.shape[0] - x.shape[0]
            pad = torch.zeros((dif, x.shape[1])).to(x.device)
            x = torch.cat((x, pad), dim=0)

        # residual connection + update
        x = self.alpha * x + (1-self.alpha)*x0
        x = self.W(x)

        return x, h

    def reset_parameters(self):
        self.V2EConvs.reset_parameters()
        self.E2VConvs.reset_parameters()
        if self.mlp3_layers > 0:
            self.W.reset_parameters()

class SetLayer(nn.Module):
    """
    One stage of WHNN. It can be either
        node-to-edge or edge-to-node
        and it contains both the encoder and the aggregate step

        The encoder can be edge-independent(MLP, Id) or edge-dependent (SAB or ISAB)
        The pooling is either standard (DeepSet or PMA) or our Wasserstein (FPSWE or LPSWE)
    """
    def __init__(self, d_in, d_out, num_layers, d_hid, proc_type, pooling_type, args):
        super(SetLayer, self).__init__()
        self.pooling_type = pooling_type
        if proc_type == 'MLP':
            # This is an edge-independent encoder: node-wise MLP
            self.proc = MLP_DS(d_in = d_in, 
                                d_out = d_out, 
                                num_layers = num_layers, 
                                hidden_layer_size = d_hid,
                                norm_type=args.normalization,
                                args = args)
            pool_d_in = d_out
        elif proc_type == 'Id':
            # This means no edge encoder
            self.proc = Identity()
            pool_d_in = d_in
        elif proc_type == 'SAB':
            # This is an edge-dependent encoder (SAB)
            ln = (args.normalization=='ln')
            self.proc = SAB(d_in, d_out, args.heads, ln=ln)
            pool_d_in = d_out
        
        elif proc_type == 'ISAB':
            # This is the more effiicent edge-dependent encoder (ISAB)
            ln = (args.normalization=='ln')
            self.proc = ISAB(d_in, d_out, args.heads, num_inds=args.isab_num_inds, ln=ln)
            pool_d_in = d_out
        
        if pooling_type == 'DeepSet':
            # this is the standard DeepSet
            self.pooling = GAP(use_mlp = True,
                                d_in = pool_d_in, 
                                d_out = d_out, 
                                num_layers = num_layers, 
                                hidden_layer_size = d_hid,
                                norm_type=args.normalization,
                                args=args)

        elif pooling_type == 'FPSWE':
            # this is the one with fixed reference set
            self.pooling = FPSWE_pool(d_in = pool_d_in, 
                                num_anchors = args.apprepset_n_anchors, 
                                num_projections = d_out,
                                anch_freeze=True)
        elif pooling_type == 'LPSWE':
            # this is the one with learnable reference set
            self.pooling = FPSWE_pool(d_in = pool_d_in, 
                                num_anchors = args.apprepset_n_anchors, 
                                num_projections = d_out,
                                anch_freeze=False)

        elif pooling_type == 'PMA':
            # this is the standard Self-Transformer (PMA)
            self.pooling = PMAPool(pool_d_in, d_hid,
                                d_out, num_layers, heads=args.heads, 
                                dropout=args.dropout)
                  

    def forward(self, x, h, edge_index, data, name):
        # Apply the encoder to project the features in a better space
        x, _ = self.proc(x, edge_index, data, name)
        x_inp = x
        x = x_inp
        # Apply the aggregator to pool the information
        x = self.pooling(x, edge_index, data, name)
        return x

    def reset_parameters(self):
        self.proc.reset_parameters()
        self.pooling.reset_parameters()




