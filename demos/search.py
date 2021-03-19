#%%
import numpy as np
import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.compute.distances import DistanceType 
from shapelets.data import load_mat, load_dataset 

data = load_mat('ItalianPowerDemand.mat')[0:15000,2]
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data)
plt.show()


# %%
query = load_mat('ItalianPowerDemand.mat')[15000:15200,2]
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
ax.plot(data[pos:pos+200])
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
    ax.plot(data[i:i+200], alpha=0.2)
ax.plot(query)
plt.show()

fig, ax = plt.subplots(figsize=(18, 8))
for i in np.array(indices[-10:]):
    ax.plot(data[i:i+200], alpha=0.2)
ax.plot(query)
plt.show()


# %%

## Running multiple queries at once!
sample_region = load_mat('ItalianPowerDemand.mat')[15000:16000,2]
queries = sc.unwrap(sample_region, 200, 1, 200, 1)
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
    start = np.array(indices[i])[0]
    end = start + 200
    ax[i].plot(data[start:end])
    ax[i].plot(queries[:, i])
plt.show()

# %%

# Visualizing Distances
sc.distances.mpdist(queries[:, 0], queries[:, 1], w=20)

# %%
labels=["Ts1", "Ts2", "Ts3", "Ts4", "Ts5"]
dst = sc.distances.pdist(queries, DistanceType.MPDist, w=25)
# %%

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
dists = squareform(dst)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels)
plt.show()
# %%

# With SBD Distance
dists = squareform(sc.distances.pdist(queries, DistanceType.SBD))
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels)
plt.show()

# %%

# With Euclidian Distance
dists = squareform(sc.distances.pdist(queries, DistanceType.Euclidean))
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels)
plt.show()

# %%
