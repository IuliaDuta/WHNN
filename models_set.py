import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling_set import *
from equiv_set import *
from layers_set import *


class SetHNN(nn.Module):
    """
    This is the main hypergraph model class which 
    run a sequence of HyperLayer(s) followed
    by the MLP classifier.

    Nothing special here.
    """
    def __init__(self, args, norm=None):
        super(SetHNN, self).__init__()
        self.All_num_layers = args.All_num_layers # the number of mpnn layers
        num_features = args.num_features
        self.aggr = args.aggregate 
        self.NormLayer = args.normalization # the type of normalisation used inside the layers
        self.sharing = args.sharing # sharing or not the params between layers

        self.layers = nn.ModuleList()
        self.dropout = args.dropout # dropout inside
        self.input_dropout = args.input_dropout # dropout on the input
     
        # While the code technically allows you to use different encoders/poolings
        # for the 2 stages, all experiments share them for simplicity

        self.proc_type_V2E = args.proc_type # the type of encoder used for V2E
        self.proc_type_E2V = args.proc_type # the type of encoder used for E2V

        self.pooling_type_V2E = args.pooling_type # the type of pooling used for V2E
        self.pooling_type_E2V = args.pooling_type # the type of pooling used for E2V

        self.lin = nn.Linear(num_features, args.MLP_hidden)
        self.layers.append(HyperLayer(proc_type_V2E = self.proc_type_V2E,
                                        pooling_type_V2E = self.pooling_type_V2E,
                                        proc_type_E2V = self.proc_type_E2V,
                                        pooling_type_E2V = self.pooling_type_E2V,
                                        args=args))

        
        for _ in range(self.All_num_layers-1):
            self.layers.append(HyperLayer(proc_type_V2E = self.proc_type_V2E,
                                        pooling_type_V2E = self.pooling_type_V2E,
                                        proc_type_E2V = self.proc_type_E2V,
                                        pooling_type_E2V = self.pooling_type_E2V,
                                        args=args))

        self.classifier = MLP(in_channels=args.MLP_hidden,
                            hidden_channels=args.Classifier_hidden,
                            out_channels=args.num_classes,
                            num_layers=args.Classifier_num_layers,
                            dropout=self.dropout,
                            Normalization=self.NormLayer,
                            InputNorm=False)

    def reset_parameters(self):
        self.lin.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

        self.classifier.reset_parameters()


    def forward(self, data):
        '''
            standard application of self.layers WHNN layers,
            for node-level classification
        '''
        x, edge_index = data.x, data.edge_index

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        x = F.dropout(x, p=self.input_dropout, training=self.training) # Input dropout
        x = self.lin(x)
        x0 = x.clone()
        for i, _ in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            idx = 0 if self.sharing else i
            x, _ = self.layers[idx](x, x0, edge_index, reversed_edge_index, data)
            x = F.relu(x)

        split_idx_dict = None

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x).squeeze(-1) 
        return x, split_idx_dict














