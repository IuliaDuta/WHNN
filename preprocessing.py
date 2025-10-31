import torch

import numpy as np

from collections import Counter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
import pdb

def RemoveIsolated(data):
	edge_index = data.edge_index

	num_nodes = data.x.shape[0]
	active_nodes = torch.unique(edge_index[0])
	new_num_nodes = active_nodes.size(0)

	old_to_new = -torch.ones(num_nodes, dtype=torch.long)
	old_to_new[active_nodes] = torch.arange(new_num_nodes)

	edge_index_remapped = edge_index.clone()
	edge_index_remapped[0] = old_to_new[edge_index[0]]
	x = data.x[active_nodes]
	y = data.y[active_nodes]

	data.x = x
	data.y = y
	data.edge_index = edge_index_remapped
	data.n_x = new_num_nodes

	return data

def extend_edge_index(edge_index):
	hyperedge_index = edge_index
	elements = hyperedge_index[0]
	indexes = hyperedge_index[1]

	incidence_id = torch.arange(hyperedge_index.shape[1]).to(elements.device)

	unique_groups = indexes.unique()
	grouped_pairs = []
	for g in unique_groups:
		group_elements = elements[indexes == g]
		group_incidence_id = incidence_id[indexes == g]
		if len(group_elements) > 0:
			pairs = torch.cartesian_prod(group_elements, group_elements)		
			group_incidence_id = torch.cartesian_prod(group_incidence_id, group_incidence_id)
			group_incidence_id = group_incidence_id[:,0].unsqueeze(-1)

			big_index = torch.ones_like(pairs[:,0]).unsqueeze(-1) * g
			pairs = torch.cat((pairs, big_index,group_incidence_id), dim=-1)
			grouped_pairs.append(pairs)
	# this will be M x 4
	extended_index = torch.cat(grouped_pairs) if grouped_pairs else torch.empty((0, 4), dtype=torch.long)
	return extended_index.transpose(0,1)


def ExtractV2E(data):
	# Assume edge_index = [V|E;E|V]
	edge_index = data.edge_index
	# First, ensure the sorting is correct (increasing along edge_index[0])
	_, sorted_idx = torch.sort(edge_index[0])
	edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

	num_nodes = data.n_x
	if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
		print('num_hyperedges does not match! 1')
		return
	cidx = torch.where(edge_index[0] == num_nodes)[
		0].min()  # cidx: [V...|cidx E...]
	data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)

	return data


def Add_Self_Loops(data):
	''' Assume edge_index = [V;E]. If not, use ExtractV2E()
		Assign a new "unique" 
		hyperedge_id for each node acts as the self-loops 
		in the standard graph case.
	'''
	edge_index = data.edge_index
	num_nodes = data.n_x
	num_hyperedges = data.num_hyperedges

	if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
		print('num_hyperedges does not match! 2')
		return

	hyperedge_appear_fre = Counter(edge_index[1].numpy())
	# store the nodes that already have self-loops
	skip_node_lst = []
	for edge in hyperedge_appear_fre:
		if hyperedge_appear_fre[edge] == 1:
			skip_node = edge_index[0][torch.where(
				edge_index[1] == edge)[0].item()]
			skip_node_lst.append(skip_node.item())

	skip_node_lst = list(set(skip_node_lst))
	new_edge_idx = edge_index[1].max() + 1
	new_edges = torch.zeros(
		(2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)

	tmp_count = 0
	for i in range(num_nodes):
		if i not in skip_node_lst:
			new_edges[0][tmp_count] = i
			new_edges[1][tmp_count] = new_edge_idx
			new_edge_idx += 1
			tmp_count += 1

	data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
	edge_index = torch.cat((edge_index, new_edges), dim=1)
	# Sort along w.r.t. nodes
	_, sorted_idx = torch.sort(edge_index[0])
	data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
	return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
	if TYPE == 'V2E':
		if option == 'all_one':
			data.norm = torch.ones_like(data.edge_index[0])

		elif option == 'deg_half_sym':
			edge_weight = torch.ones_like(data.edge_index[0])
			cidx = data.edge_index[1].min()
			Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
			HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
			V_norm = Vdeg**(-1/2)
			E_norm = HEdeg**(-1/2)
			data.norm = V_norm[data.edge_index[0]] * \
				E_norm[data.edge_index[1]-cidx]

	elif TYPE == 'V2V':
		data.edge_index, data.norm = gcn_norm(
			data.edge_index, data.norm, add_self_loops=True)
	return data


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
	""" Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
	""" randomly splits label into train/valid/test splits """
	if not balance:
		if ignore_negative:
			labeled_nodes = torch.where(label != -1)[0]
		else:
			labeled_nodes = label

		n = labeled_nodes.shape[0]
		train_num = int(n * train_prop)
		valid_num = int(n * valid_prop)

		perm = torch.as_tensor(np.random.permutation(n))

		train_indices = perm[:train_num]
		val_indices = perm[train_num:train_num + valid_num]
		test_indices = perm[train_num + valid_num:]

		if not ignore_negative:
			return train_indices, val_indices, test_indices

		train_idx = labeled_nodes[train_indices]
		valid_idx = labeled_nodes[val_indices]
		test_idx = labeled_nodes[test_indices]

		split_idx = {'train': train_idx,
					 'valid': valid_idx,
					 'test': test_idx}
	else:

		indices = []
		for i in range(label.max()+1):
			index = torch.where((label == i))[0].view(-1)
			index = index[torch.randperm(index.size(0))]
			indices.append(index)

		percls_trn = int(train_prop/(label.max()+1)*len(label))
		val_lb = int(valid_prop*len(label))
		train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
		rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
		rest_index = rest_index[torch.randperm(rest_index.size(0))]
		valid_idx = rest_index[:val_lb]
		test_idx = rest_index[val_lb:]
		split_idx = {'train': train_idx,
					 'valid': valid_idx,
					 'test': test_idx}
	return split_idx


