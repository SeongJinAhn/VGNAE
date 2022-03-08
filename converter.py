import os.path as osp
import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data


def read_data_from_npz(path: str, filename: str):
    origin = np.load('%s/%s.npz' % (path, filename))
    adj_coo_matrix = ssp.csc_matrix(
        (origin['adj_matrix.data'], origin['adj_matrix.indices'], origin['adj_matrix.indptr'])).tocoo()
    attr_matrix = ssp.csc_matrix(
        (origin['attr_matrix.data'], origin['attr_matrix.indices'], origin['attr_matrix.indptr'])).toarray().transpose()
    edge_index = torch.tensor([adj_coo_matrix.row, adj_coo_matrix.col], dtype=torch.int64)
    x = torch.from_numpy(attr_matrix)
    return [Data(x=x, edge_index=edge_index)]


if __name__ == '__main__':
    data = read_data_from_npz(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'fb_1684'), 'fb_1684')
    print(data[0])
