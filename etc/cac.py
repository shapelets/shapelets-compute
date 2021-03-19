#%%
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt 
import shapelets.compute as sc 
import cProfile

current_path = pathlib.Path(__file__).parent.absolute()
file = os.path.join(current_path, "regime.txt")
data = np.loadtxt(file)

# plt.plot(data)
# plt.show()

# %%
profile, index = sc.matrixprofile.matrix_profile(data, 100)
# plt.plot(profile)
# plt.show()

# %%
cac = sc.matrixprofile.cac(profile, index, 100)
# print(sc.argmin(cac))
# plt.plot(cac)
# plt.show()

cProfile.run('sc.matrixprofile.matrix_profile(data, 100)')
cProfile.run('sc.matrixprofile.cac(profile, index, 100)')






# %%

# pos = sc.range(len(index), dtype=index.dtype)
# small = sc.minimum(pos, index)
# small_mark = sc.zeros(len(index), dtype=index.dtype)
# large = sc.maximum(pos, index)
# large_mark = sc.zeros(len(index), dtype=index.dtype)

# # %%
# xx = sc.sort(small)
# si, sv = sc.sum_by_key(xx, sc.ones(xx.shape, dtype=index.dtype))
# small_mark.assign(si, sv)
# print(small_mark.shape[0] == index.shape[0])

# xx = sc.sort(large)
# li, lv = sc.sum_by_key(xx, sc.ones(xx.shape, dtype=index.dtype))
# large_mark.assign(li, lv)
# print(large_mark.shape[0] == index.shape[0])
# mark = small_mark - large_mark


# # %%
# crosscount = sc.cumsum(mark)

# # %%
# plt.plot(crosscount)
# plt.show()
# # %%
# l = crosscount.shape[0]
# i = sc.iota(l)
# adj = 2*i*(l-i)/l

# # %%
# plt.plot(adj)
# plt.show()
# # %%
# normalized_crosscount = sc.minimum(crosscount / adj, 1)

# # %%
# plt.plot(normalized_crosscount)
# plt.show()
# # %%
