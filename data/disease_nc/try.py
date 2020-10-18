import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

edges = open("disease_nc.edges.csv", "r")
print(len(edges.readlines()))
labels = np.load(os.path.join(".", "disease_nc.labels.npy"))
# print(edges.readlines()[0])
print(labels)
