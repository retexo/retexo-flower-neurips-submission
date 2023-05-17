import torch
import os.path as osp
import pickle
import sys
import warnings
import itertools
import numbers
import scipy.sparse as sparse
import numpy as np
import networkx as nx
from torch import Tensor
from copy import copy
from collections import defaultdict

def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_index = test_index.sort()[0]    
    
    x = torch.cat([allx, tx], dim=0)
    x[test_index] = x[sorted_test_index]
    
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
    
    return x, y, train_mask, val_mask, test_mask, edge_index    
    
def read_file(folder, prefix, name):
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            warnings.filterwarnings('ignore', '.*`scipy.sparse.csr` name.*')
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.from_numpy(out).to(torch.float)
    return out

def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = int
    if torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float

    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src).to(dtype).squeeze()
    return src

def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)

def index_to_mask(index, size = None):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def repeat(src, length):
    if src is None:
        return None
    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src

def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index

def remove_self_loops(
    edge_index,
    edge_attr = None,
):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]
    
def is_torch_sparse_tensor(src):
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
    return False

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def coalesce(
    edge_index,
    edge_attr = '???',
    num_nodes = None,
    is_sorted = False,
    sort_by_row = True,
):

    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        # idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, (Tensor, list, tuple)):
            return edge_index, edge_attr
        return edge_index

    edge_index = edge_index[:, mask]

    dim_size = None
    if isinstance(edge_attr, (Tensor, list, tuple)):
        dim_size = edge_index.size(1)
        idx = torch.arange(0, nnz, device=edge_index.device)
        idx.sub_(mask.logical_not_().cumsum(dim=0))

    if edge_attr is None:
        return edge_index, None

    return edge_index

def augNormGCN(adj):
    adj += sparse.eye(adj.shape[0])
    
    degree_for_norm = sparse.diags(
        np.power(np.array(adj.sum(1)), -0.5).flatten()
    )
    adj_hat_csr = degree_for_norm.dot(
        adj.dot(degree_for_norm)
    )
    
    adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)
    return adj_hat_csr, adj_hat_coo

def get_adjacency_matrix(edge_index):
    adj = to_scipy_sparse_matrix(edge_index)
    _, adj_hat_coo = augNormGCN(adj)
    
    indices = torch.from_numpy(
        np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64)
    )    
    
    values = torch.from_numpy(adj_hat_coo.data)    

    adjacency_matrix = torch.sparse_coo_tensor(
        indices, values, torch.Size(adj_hat_coo.shape)
    )
    
    return adjacency_matrix

def to_scipy_sparse_matrix(edge_index):
    row, col = edge_index.cpu()
    edge_attr = torch.ones(row.size(0))
    
    N = maybe_num_nodes(edge_index)
    out = sparse.coo_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out

def to_networkx(data, node_attrs = None, to_undirected = False, remove_self_loops=False):
    G = nx.Graph() if to_undirected else nx.DiGraph()
    
    G.add_nodes_from(range(data["x"].size(1)))
    
    values = {}
    
    for key in node_attrs:
        value = data[key]
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value
    
    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False
    
    for i, (u, v) in enumerate(data["edge_index"].t().tolist()):
        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    
    return G

def from_networkx(G, group_node_attrs=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)

    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
        
    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)
    
    for key, value in G.graph.items():
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value
    
    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                pass
            
    data['edge_index'] = edge_index.view(2, -1)
    
    return data

def dataset_partitioner(data, client_id, number_of_clients):
    np.random.seed(123)
    dataset_size = data["x"].size(0)
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    
    np.random.shuffle(dataset_indices)
    
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True
    )
    
    nx.set_node_attributes(G,dict([(nid, nid) for nid in range(nx.number_of_nodes(G))]), name="index_orig")
    
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    
    nodes = set(dataset_indices[start_ind:end_ind])
    first_hop_nodes = copy(nodes)
    
    for node in nodes:
        first_hop_nodes.update(list(nx.all_neighbors(G, node)))
        
    nodes = torch.tensor(list(nodes))
    d = from_networkx(nx.subgraph(G, first_hop_nodes))
    
    first_hop_nodes = torch.tensor(list(first_hop_nodes))
    
    d['nodes'] = nodes
    d['first_hop_nodes'] = first_hop_nodes
    return d

def node_indices(data, values):
    indices = torch.tensor([], dtype=torch.int64)
    
    for num in values:
        idx = torch.where(data["index_orig"] == num)[0]
        indices = torch.cat((indices, idx), dim=0)
    
    return indices
        
def load_data(data_dir, dataset, cid, num_clients):
    x, y, train_mask, val_mask, test_mask, edge_index = read_planetoid_data(data_dir, dataset)
    data = {
        'x': x,
        'y': y,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'edge_index': edge_index
    }
    
    client_data = dataset_partitioner(data, cid, num_clients)
    nodes = torch.zeros(client_data["train_mask"].size(), dtype=torch.bool)
    first_hop_nodes = torch.zeros(client_data["train_mask"].size(), dtype=torch.bool)

    nodes[node_indices(client_data, client_data["nodes"])] = True
    first_hop_nodes[node_indices(client_data, client_data["first_hop_nodes"])] = True
    
    client_data["nodes"] = nodes
    client_data["first_hop_nodes"] = first_hop_nodes

    client_data["train_mask"] = torch.logical_and(client_data["train_mask"], nodes)
    client_data["val_mask"] = torch.logical_and(client_data["val_mask"], nodes)
    client_data["test_mask"] = torch.logical_and(client_data["test_mask"], nodes)
    
    client_data["x"][torch.logical_not(client_data["nodes"]), :] = 0
    client_data["adj"] = get_adjacency_matrix(client_data["edge_index"])
    return client_data
    
if __name__ == "__main__":
    client_data = load_data("data", "cora", 0, 2)