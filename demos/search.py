#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
import numpy as np
import matplotlib.pyplot as plt
import shapelets.compute as sc
from shapelets.data import load_mat, load_dataset

import warnings

warnings.filterwarnings("ignore")

data = load_mat('ItalianPowerDemand.mat')[0:15000, 2]
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data)
plt.show()

# %%
query = load_mat('ItalianPowerDemand.mat')[15000:15200, 2]
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(query)
plt.show()

# %%
# Search closest match
single_search = sc.matrixprofile.mass(query, data)

# %%
pos, _ = sc.argmin(single_search)

# %%
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data[pos:pos + 200])
ax.plot(query)
plt.show()

# %%
indices, values = sc.sort_index(single_search)
print("Closest...")
for i in np.array(indices[:10]):
    print(i)

print("\nQuite different...")
for i in np.array(indices[-10:]):
    print(i)

# %%
fig, ax = plt.subplots(figsize=(18, 8))
for i in np.array(indices[:10]):
    ax.plot(data[i:i + 200], alpha=0.2)
ax.plot(query)
plt.show()

fig, ax = plt.subplots(figsize=(18, 8))
for i in np.array(indices[-10:]):
    ax.plot(data[i:i + 200], alpha=0.2)
ax.plot(query)
plt.show()

# %%
## Running multiple queries at once!
sample_region = load_mat('ItalianPowerDemand.mat')[15000:16000, 2]
queries = sc.unpack(sample_region, 200, 1, 200, 1)
print(queries.shape)
fig, ax = plt.subplots(5, 1, figsize=(18, 8))
for i in range(5):
    ax[i].plot(queries[:, i])
plt.show()

# %%
multiple_results = sc.matrixprofile.mass(queries, data)

# %%
indices, distances = sc.argmin(multiple_results, 0)
fig, ax = plt.subplots(5, 1, figsize=(18, 8))
for i in range(5):
    start = indices[0, i]
    end = start + 200
    ax[i].plot(data[start:end])
    ax[i].plot(queries[:, i])
plt.show()

# %%
# Visualizing Distances
sc.distances.mpdist(queries[:, 0], queries[:, 1], w=20)

# %%
l = ["Ts1", "Ts2", "Ts3", "Ts4", "Ts5"]
dst = sc.distances.pdist(queries, 'mpdist', w=25)

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

dists = squareform(dst)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=l)
plt.show()

# %%
# With SBD Distance
dists = squareform(sc.distances.pdist(queries, 'sbd'))
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=l)
plt.show()

# %%
# With Euclidian Distance
dists = squareform(sc.distances.pdist(queries, 'euclidean'))
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=l)
plt.show()
