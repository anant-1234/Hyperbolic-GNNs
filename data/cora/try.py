
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
# import torch


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))

    # test_idx_range = np.sort(test_idx_reorder)

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]

    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # labels = np.argmax(labels, 1)

    # idx_test = test_idx_range.tolist()
    # idx_train = list(range(len(y)))
    # idx_val = range(len(y), len(y) + 500)
    # print(graph, graph.number_of_nodes(), graph.number_of_edges())
    t = nx.from_dict_of_lists(graph)    
    print(t.number_of_nodes(), t.number_of_edges())

    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # if not use_feats:
    #     features = sp.eye(adj.shape[0])
    # return adj, features, labels, idx_train, idx_val, idx_test

load_citation_data("cora", 0, ".")