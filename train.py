#!/usr/bin/env python
# coding: utf-8
import os
import time
import torch
import argparse

import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import Logger
from preprocessing import *

from convert_datasets_to_pygDataset import dataset_Hypergraph

from models_set import *
import time


os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "200"

np.random.seed(0)
torch.manual_seed(0)

torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None, final_epoch=False):
    if result is not None:
        out = result
    else:
        model.eval()
        out, pair_split_idx = model(data)

        if pair_split_idx is not None:
            split_idx = pair_split_idx
    
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']], name='train')
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']], name='valid')
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']], name='test')


    train_loss = F.nll_loss(out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])

    train_conf_matrix = None
    valid_conf_matrix = None
    test_conf_matrix = None

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out, \
                    (train_conf_matrix, valid_conf_matrix, test_conf_matrix)


def eval_acc(y_true, y_pred, name):
    acc_list = []
    y_true = y_true
    y_pred = y_pred.argmax(dim=-1, keepdim=False)

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    if len(correct) == 0:
        acc_list.append(0.0)
    else:    
        acc_list.append(float(torch.sum(correct).item())/len(correct))
    
    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""

"""

if __name__ == '__main__':
    start_time = time.time()
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--runs', default=10, type=int)  # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float) # Dropout rate
    parser.add_argument('--input_dropout', default=0.0, type=float) # Input dropout rate
    parser.add_argument('--lr', default=0.001, type=float) # Learning rate
    parser.add_argument('--wd', default=0.0, type=float) # weight decay

    parser.add_argument('--All_num_layers', default=2, type=int) # number of layers
    parser.add_argument('--MLP_num_layers', default=1, type=int)  # number of layers for the MLPs used inside the models
    parser.add_argument('--MLP_hidden', default=64, type=int)  # hidden dimension
    parser.add_argument('--Classifier_num_layers', default=1, type=int)  # number of layers inside the classififer MLP
    parser.add_argument('--Classifier_hidden', default=64, type=int)  # hidden dimension for the classifier MLP
    parser.add_argument('--display_step', type=int, default=99) 
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])

    parser.add_argument('--normtype', default='all_one') # ['all_one','deg_half_sym'] the type of norm precomputed (Not used?)
    parser.add_argument('--add_self_loop', type=str2bool, default=True) # enriching structure with self loops
    parser.add_argument('--remove_isolated', type=str2bool, default=False) # if to remove isolated nodes

    parser.add_argument('--normalization', default='ln') # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--deepset_input_norm', type=str2bool, default = False) 

    parser.add_argument('--num_features', default=0, type=int)  # number of input features
    parser.add_argument('--num_classes', default=0, type=int)  # number of classes
    
    parser.add_argument('--feature_noise', default='1', type=str) # choose std for synthetic feature noise
    parser.add_argument('--exclude_self', action='store_true') # whether the he contain self node or not

    parser.add_argument('--heads', default=1, type=int)  # number of heads in SAB
    parser.add_argument('--wandb', default=True, type=str2bool) # if logging into wandb

    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3') # number of layers for the mpnn update 
    parser.add_argument('--restart_alpha', default=0.5, type=float) # coefficient for the residual connection

   
    parser.add_argument('--tag', type=str, default='testing') #helper for wandb in order to filter out the testing runs. if set to testing we are in dev mode
    parser.add_argument('--AllSet_input_norm', type=str2bool, default=True) # used?
    
    # specific to WHNN
    parser.add_argument('--proc_type', default='MLP', type=str)  # ['MLP','Id', 'SAB', 'ISAB'] the type of encoder.
    parser.add_argument('--pooling_type', default='DeepSet', type=str) # ['DeepSet', 'PMA', 'FPSWE', 'LPSWE'] the type of pooling. 
                                                                       #        FPSWE correspond to fixed reference point while LPSWE is learnable reference

    parser.add_argument('--apprepset_n_anchors', default=5, type=int)  # number of reference points
    parser.add_argument('--isab_num_inds', default=5, type=int)  # number of anchors inside isab approx

    parser.add_argument('--sharing', type=str2bool, default=False)  # sharing or not the params between mpnn layers


    parser.set_defaults(exclude_self=False)
    

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    #     Use the line below for .py file
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(sync_tensorboard=False, project='encapsulation', reinit = False, config = args, entity='hyper_graphs', tags=[args.tag])
        print('Monitoring using wandb')
    
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40',
                        'NTU2012', 
                        'coauthor_cora', 'coauthor_dblp',
                        'house-committees',
                        'house-committees-100',
                        'cora', 'citeseer', 'pubmed', 
                        'congress-bills', 'senate-committees', 
                        'senate-committees-100', 'congress-bills-100']

    synthetic_list = ['house-committees', 'house-committees-100', 'congress-bills', 
                        'senate-committees', 'senate-committees-100', 'congress-bills-100']

    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes

        if args.dname == 'ModelNet40':
            data.y = data.y - data.y.min()
            args.num_classes = max(dataset.y).item()+1

        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
        assert data.y.min().item() == 0    
   
    # Set the hypergraph data in the incidence form
    data = ExtractV2E(data)

    # Preprocess the hypergrapj 
    if args.remove_isolated:
        data = RemoveIsolated(data)

    if args.add_self_loop:
        data = Add_Self_Loops(data)
    
    # Rename the indices to start from 0
    data.edge_index[1] -= data.edge_index[1].min()
    # Create edge-specific index used in the edge-dependent encoder
    data.extended_index = extend_edge_index(data.edge_index)
    # Same as above but used for the edge-to-node
    reversed_edge_index = torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)
    data.reversed_extended_index = extend_edge_index(reversed_edge_index)
    
    # Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)   
    data = norm_contruction(data, option=args.normtype)

    # Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
                data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
            
        split_idx_lst.append(split_idx)
    
    # Create the model
    model = SetHNN(args)

    num_params = count_parameters(model)
    print('Number of parameters', num_params)

    # Put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')  
    model, data = model.to(device), data.to(device)

    if args.wandb:
        wandb.watch(model)
    
    # Train the model
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    model.train()

    ### Training loop ###
    runtime_list = []

    train_accs_runs, valid_accs_runs, test_accs_runs = [], [], []
    train_loss_runs, valid_loss_runs, test_loss_runs = [], [], []


    for run in tqdm(range(args.runs)):
        split_idx = split_idx_lst[run]
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best_val = float('-inf')
        train_accs_one_run, valid_accs_one_run, test_accs_one_run = [], [], []
        train_loss_one_run, valid_loss_one_run, test_loss_one_run = [], [], []

        for epoch in range(args.epochs): 
            train_idx = split_idx['train'].to(device) 
            model.train()
            optimizer.zero_grad()

            out, _ = model(data)
            out = F.log_softmax(out, dim=1)

            loss = criterion(out[train_idx].double(), data.y[train_idx])
            loss.backward()

            if epoch % 30 == 0:
                num_params = count_parameters(model)
                print(f"number of parameters {num_params}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"[Epoch {epoch}] Gradient for {name}, shape {param.grad.shape} : {param.grad.abs().mean()}")
                    else:
                        print(f"[Epoch {epoch}] No gradient for {name}")

            optimizer.step()
   
            final_epoch = (epoch == args.epochs-1)
            result = evaluate(model, data, split_idx, eval_func, final_epoch=final_epoch)
            logger.add_result(run, result[:3])
    
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
                print("Train confusion\n", result[7][0])
                print("Valid confusion\n", result[7][1])
                print("Test confusion\n", result[7][2])

            if args.wandb:
                train_accs_one_run.append(100 * result[0])
                valid_accs_one_run.append(100 * result[1])
                test_accs_one_run.append(100 * result[2])
                train_loss_one_run.append(loss.item())
                valid_loss_one_run.append(result[4].item())
                test_loss_one_run.append(result[5].item())

        if args.wandb:
            # add training statistics from the crt running
            train_accs_runs.append(train_accs_one_run)
            valid_accs_runs.append(valid_accs_one_run)
            test_accs_runs.append(test_accs_one_run)
            train_loss_runs.append(train_loss_one_run)
            valid_loss_runs.append(valid_loss_one_run)
            test_loss_runs.append(test_loss_one_run)

    
    ### Save results ###
    if args.wandb:
        train_accs_runs = np.array(train_accs_runs)
        valid_accs_runs = np.array(valid_accs_runs)
        test_accs_runs = np.array(test_accs_runs)
        train_loss_runs = np.array(train_loss_runs)
        valid_loss_runs = np.array(valid_loss_runs)
        test_loss_runs = np.array(test_loss_runs)

        train_accs_mean = np.mean(train_accs_runs, 0)
        train_accs_std = np.std(train_accs_runs, 0)
        valid_accs_mean = np.mean(valid_accs_runs, 0)
        valid_accs_std = np.std(valid_accs_runs, 0)
        test_accs_mean = np.mean(test_accs_runs, 0)
        test_accs_std = np.std(test_accs_runs, 0)
        train_loss_mean = np.mean(train_loss_runs, 0)
        train_loss_std = np.std(train_loss_runs, 0)
        valid_loss_mean = np.mean(valid_loss_runs, 0)
        valid_loss_std = np.std(valid_loss_runs, 0)
        test_loss_mean = np.mean(test_loss_runs, 0)
        test_loss_std = np.std(test_loss_runs, 0)
        


        best_accuracy = -100
        for epoch in range(len(train_accs_mean)):
            best_accuracy = max(best_accuracy, valid_accs_mean[epoch])
            log_corpus = {
                f'train_accs_mean': train_accs_mean[epoch],
                f'val_accs_mean': valid_accs_mean[epoch],
                f'test_accs_mean': test_accs_mean[epoch],
                f'best_accs_mean': best_accuracy,

                f'train_acc_std': train_accs_std[epoch],
                f'test_acc_std': valid_accs_std[epoch],
                f'val_acc_std': test_accs_std[epoch],

                f'train_loss_mean': train_loss_mean[epoch],
                f'val_loss_mean': valid_loss_mean[epoch],
                f'test_loss_mean': test_loss_mean[epoch],

                f'train_loss_std': train_loss_std[epoch],
                f'test_loss_std': valid_loss_std[epoch],
                f'val_loss_std': test_loss_std[epoch],
            }
            wandb.log(log_corpus, step=epoch)

    best_val, best_test = logger.print_statistics()
    res_root = 'hyperparameter_tuning'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'SetHNN_{args.lr}_{args.wd}_{args.heads}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print('All done! Exit python code')
    quit()
    

