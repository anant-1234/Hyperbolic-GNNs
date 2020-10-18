import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

edges = open("disease_lp.edges.csv", "r")
print(len(edges.readlines()))
labels = np.load(os.path.join(".", "disease_lp.labels.npy"))
features = sp.load_npz("disease_lp.feats.npz")
# print(edges.readlines()[0])
print(features.shape)
print(len(labels))
