#%%
import shapelets.compute as sc
import shapelets.generators as sg 
import numpy as np 
import matplotlib.pyplot as plt 
import matrixprofile as morg


# %%
ts = np.array([1., 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1,  1, 2, 2, 4, 5, 1, 1, 9])
query = np.array([0.23595094, 0.9865171, 0.1934413, 0.60880883, 0.55174926, 0.77139988, 0.33529215, 0.63215848])
w = 4
ref = morg.algorithms.mpdist(ts,query,w)
ref
# %%

abp, abi = sc.matrix_profile(ts, w, query)
bap, bai = sc.matrix_profile(query, w, ts)

# %%
bai
# %%
abba = sc.join([abp, bap])
# %%
data_len = len(abp) + len(bap)
# %%
# %%
abba_sorted, _ = sc.sort(abba)
# %%
abba_sorted
# %%
abba
# %%
upper_idx = int(np.ceil(0.05*data_len))-1

# %%
idx = np.min([len(abba_sorted) - 1, upper_idx])
# %%
abba_sorted[idx]
# %%
idx
# %%
abba_sorted[1:1]
# %%
abba_sorted
# %%
ref
# %%
abba_sorted[0:1]
# %%
abba_sorted[idx-1:idx]
# %%
t = np.array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]])   
# %%
t.shape
# %%
len(t)
# %%
