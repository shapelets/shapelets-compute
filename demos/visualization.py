#%%

import numpy as np
import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.compute.distances import DistanceType 
from shapelets.data import load_mat, load_dataset 

import warnings
warnings.filterwarnings("ignore")
#%%
# data = load_mat('ItalianPowerDemand.mat')[:,3]
# data = load_dataset("regime.txt")
data = load_dataset('ecg-heartbeat-av.txt')[32:]
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data)
plt.show()


#%%
dataidx = sc.iota(data.size, dtype=data.dtype)
reduced = sc.dimensionality.visvalingam(dataidx, data, 1000)
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(reduced[:,0], reduced[:,1])
plt.show()


# r = sc.dimensionality.paa(dataidx, data, 200)
# plt.plot(r[:,0], r[:,1])
# plt.show()




# %%
gb = 128 
hour_day = sc.unpack(data, gb, 1, gb, 1)
plt.imshow(hour_day, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# %%
r = sc.convolve1(hour_day, [1, -2, 1.], 'default')
plt.imshow(r, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()
# %%
plt.imshow(sc.diff2(hour_day), cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# %%
plt.imshow(sc.diff1(hour_day), cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()
# %%

# %%
svd = sc.svd(hour_day)
low_rank = svd.low_rank(3)
plt.imshow(low_rank, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

reconstructed = sc.pack(low_rank, low_rank.size, 1, gb, 1, gb, 1)
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(reconstructed)
plt.show()
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data[0:reconstructed.shape[0]] - reconstructed)
plt.show()

# %%
svd.acc_pct
# %%
plt.plot(svd.u[:,0])
plt.plot(svd.u[:,1])
plt.show()
plt.plot(svd.vt[0,:].T)
plt.plot(svd.vt[1,:].T)
plt.show()
# %%
svd.pct
# %%
import matplotlib.pyplot as plt
import shapelets.compute as sc
import numpy as np
r = sc.random.random_engine()
x = r.gamma(2.0, 0.5, shape=100000)
plt.hist(x, 200)
plt.grid()
plt.show()
# %%
