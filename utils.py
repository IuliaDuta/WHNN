import torch, math, numpy as np, scipy.sparse as sp

import numpy as np
import time

from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame

import pdb
import wandb
import time

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            wandb.log({"final_test_accs" : r.mean(), "final_test_accs_std" : r.std()})

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])

def sparse_sort(src: torch.Tensor, index: torch.Tensor, descending=False, eps=1e-12):
    #src: big_N x d
                       
    f_src = src.float()
    f_min, f_max = f_src.min(0)[0], f_src.max(0)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) 
    norm = norm * 0.8 + 0.1
    norm = norm + index.float().unsqueeze(-1)*(-1)**int(descending)

    perm = norm.argsort(dim=0, descending=descending)


    out = src[perm, torch.arange(perm.shape[1], device=src.device)]
    # note: this now correspond to index[perm] instead of index
    return out, perm

def interp1d(x,y,xnew,ind,hedge_idx=None):
    M,N=xnew.shape
    ind = ind.clone().long()
    
    if N==1:
        torch.searchsorted(x.contiguous().squeeze(),
                               xnew.contiguous(), out=ind)
    else:
        torch.searchsorted(x.contiguous().squeeze(),
                               xnew.contiguous().squeeze(), out=ind, right=True)

    eps = 0.000001

    slopes = (y[:, 1:]-y[:, :-1])/(eps + (x[:, 1:]-x[:, :-1]))
    ind -= 1
    ind = torch.clamp(ind, 0, x.shape[1] - 1 - 1)

    if (hedge_idx[ind+1]!=hedge_idx[ind]).any() == True:
        pdb.set_trace()

    def sel(x):

        return torch.gather(x, 1, ind)
    
    ynew = sel(y) + sel(slopes)*(xnew-sel(x))
    
    return ynew.to(torch.float32)

def get_color_coded_background(color, i):
    return "\033[4{}m {:.4f} \033[0m".format(color+1, i)

def print_a_colored_ndarray(map, d, row_sep=""):
    map = np.round(map,3)
    n,m = map.shape 
    n = n // d
    m = m // d
    color_range_row = np.arange(m)[np.newaxis,...].repeat(d,axis=1)
    color_range_col = np.arange(n)[...,np.newaxis].repeat(d,axis=0)
    color_range = color_range_row + color_range_col

    back_map_modified = np.vectorize(get_color_coded_background)(color_range, map)
    n, m = back_map_modified.shape
    fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
    print(fmt_str.format(*back_map_modified.ravel()))


class CollaterHyper:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            collate_batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # #changed to allow assignments of hyperedges as well
            repeats = [hgraph.num_hyperedges or 0 for hgraph in batch]
            batch_hedges = [torch.full((n, ), i) for i, n in enumerate(repeats)]
            batch_hedges =  torch.cat(batch_hedges, dim=0)
            collate_batch.batch_hedges = batch_hedges
            return collate_batch
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
