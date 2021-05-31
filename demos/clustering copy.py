#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
import shapelets.compute as sc
from numpy import loadtxt
from shapelets.data import load_dataset

# raw_test = load_dataset("Coffee_TEST.txt")
# test_classes = raw_test[:,0].T
# test_data = raw_test[:,1:].T

raw_train = load_dataset("Coffee_TRAIN.txt")
train_classes = raw_train[:, 0].T
train_data = raw_train[:, 1:].T

# %%
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

dists = squareform(sc.distances.pdist(train_data, 'sbd', w=25))
Z = hac.linkage(dists, method='centroid', metric='precomputed')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
hac.dendrogram(Z)
plt.show()
