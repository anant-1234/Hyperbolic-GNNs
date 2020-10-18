import networkx as nx 
import pickle as pkl 
import numpy as np 
import scipy.sparse as sp
import torch

graph = pkl.load(open('airport.p', 'rb'))
# graph = pkl.load(open('airport_alldata.p', 'rb'))

# alll = pkl.load(open('airport_alldata.p', 'rb'))
features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
print(features.shape, features.size)
print(features.size(0))
# print(nx.adjacency_matrix(alll))
# adj = nx.adjacency_matrix(graph)
# deg = np.squeeze(np.sum(adj, axis=0).astype(int))
# deg[deg > 5] = 5
# deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
# # print(features.size)
# print(deg_onehot, deg_onehot.shape)
# # features = torch.tensor(features)
# const_f = torch.ones(features.size, 1)
# print(const_f, const_f.shape)
# features2 = torch.cat((features, deg_onehot, const_f), dim=1)
# print(features.shape, features2.shape)
# print(graph)
# # print(adj)
# x, y = sp.triu(adj).nonzero()
# pos_edges = np.array(list(zip(x, y)))
# np.random.shuffle(pos_edges)
# # print(pos_edges.shape)
# print(adj.toarray())
# print(np.sum(adj, axis=0).astype(int))
# print(np.squeeze(np.sum(adj, axis=0).astype(int)))
# print(feats.shape)
# print(feats[:, 4].shape)
# print(feats[:, :4].shape)
# print(graph)
# print(nx.adjacency_matrix(graph))
# print(graph.number_of_nodes())
# print(graph.number_of_edges())
# w=graph.edges
# print(sorted(w))