# code adapted from https://github.com/mint-vu/backbone_vs_pooling/blob/main/poolings/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torch_geometric.utils import softmax
import math

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, data, name):
        # Directly return x but put it in the right format
        return x[edge_index[0]], x

class MLP_DS(nn.Module):
    # this is exactly MlpBlock but used it like this for output matching
    def __init__(self, d_in=None, d_out=None, num_layers=2, hidden_layer_size=512, norm_type='None', args=None):
        super().__init__()
        self.net = MLP(d_in, hidden_layer_size, d_out, num_layers, dropout=args.dropout, Normalization=norm_type, InputNorm=True)

    def forward(self, x, edge_index, data, name):
        x = self.net(x)
        return x[edge_index[0]], x
        
    def reset_parameters(self):
        self.net.reset_parameters()


class MAB(nn.Module):
    '''
        The MAB model as proposed in SAB and ISAB
    '''
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln = ln
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.fc_k.reset_parameters()
        self.fc_v.reset_parameters()
        self.fc_o.reset_parameters()
        
        if self.ln:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()


    def forward(self, Q, K,  hyperedge_index_0, extended_index):
        LE_index = extended_index[:,3]

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
     
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.stack(Q.split(dim_split, 1), 0) # H x K x D 
        K_ = torch.stack(K.split(dim_split, 1), 0) # H x N x D 
        V_ = torch.stack(V.split(dim_split, 1), 0) # H x N x D 


        Q_i = torch.index_select(Q_, index=extended_index[:,0], dim=1) # H x M x D
        K_j = torch.index_select(K_, index=extended_index[:,1], dim=1) # H x M x D
        V_j = torch.index_select(V_, index=extended_index[:,1], dim=1) # H x M x D

        #j receives from i
        A_ij = (Q_i * K_j).sum(-1)/math.sqrt(self.dim_V) # H x M 


        A_ij = softmax(A_ij, LE_index, dim=1).unsqueeze(-1) # H x M x 1

        QKV_i = scatter_add(A_ij*V_j, LE_index, dim=1) #H x m x 1
        Q_i = torch.index_select(Q_, index=hyperedge_index_0, dim=1) # H x m x D


        O_i = Q_i + QKV_i #H x m x 1
        O_i = torch.permute(O_i, (1,2,0))
        O_i = O_i.reshape(O_i.shape[0], -1) #m x (H*d)
        O_i = O_i if getattr(self, 'ln0', None) is None else self.ln0(O_i)
        O_i = O_i + F.relu(self.fc_o(O_i))
        O_i = O_i if getattr(self, 'ln1', None) is None else self.ln1(O_i)

        return O_i

class SAB(nn.Module):
    '''
        SAB model as proposed in: https://arxiv.org/pdf/1810.00825
    '''
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, X, hyperedge_index, data, name):
        # this will be M x 3
        if name == 'V2E':
            extended_index = data.extended_index.transpose(0,1)
        elif name == 'E2V':
            extended_index = data.reversed_extended_index.transpose(0,1)
      
        return self.mab(X, X, hyperedge_index[0], extended_index), None

class ISAB(nn.Module):
    '''
        ISAB model as proposed in: https://arxiv.org/pdf/1810.00825
    '''
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

        self.num_inds = num_inds
        self.inc_val = None
        self.hyperedge_index_I_X_0 = None
        self.extended_index_X_I_3 = None

    def reset_parameters(self):
        self.mab0.reset_parameters()
        self.mab1.reset_parameters()
        nn.init.xavier_uniform_(self.I)

    def forward(self, X, hyperedge_index, data, name):
        num_edges = hyperedge_index[1].max()+1
        m = hyperedge_index.shape[1]


        I_big = self.I.repeat(num_edges, 1) #num_edges*m x d
        edge_extended = hyperedge_index.repeat(1,self.num_inds)

        if self.inc_val is None:
            self.inc_val = torch.arange(self.num_inds).to(X.device)

        extended_index_1 = edge_extended[0]
        extended_index_2 = edge_extended[1] * self.num_inds + self.inc_val.repeat_interleave(m)
        extended_index_3 = extended_index_2.clone()

        extended_index_I_X = torch.stack((extended_index_2, extended_index_1, extended_index_3, extended_index_3), dim=1)
        
        if self.hyperedge_index_I_X_0 is None:
            self.hyperedge_index_I_X_0 = torch.arange(self.num_inds*num_edges).to(X.device)

        #create edge_index for it 
        H = self.mab0(I_big, X, self.hyperedge_index_I_X_0, extended_index_I_X)

        if self.extended_index_X_I_3 is None:
            self.extended_index_X_I_3 = torch.arange(m).repeat(self.num_inds).to(X.device)
        extended_index_X_I = torch.stack((extended_index_1, extended_index_2, self.extended_index_X_I_3, self.extended_index_X_I_3), dim=1)
        hyperedge_index_X_I_0 = hyperedge_index[0]

        return self.mab1(X, H, hyperedge_index_X_I_0, extended_index_X_I), None



class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        elif Normalization == 'ln':
            print("using LN")
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                # print(self.normalizations[0].device)
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            # x = F.tanh(x)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x