#%%
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt 
import shapelets.compute as sc 
from shapelets.data import load_dataset
import cProfile

current_path = pathlib.Path(__file__).parent.absolute()
file = os.path.join(current_path, "vanilla_ice.csv")

data = np.loadtxt('/Users/justo.ruiz/Development/shapelets/solo_comprobacion/modules/shapelets/data/Patient1.Mix 02.amc.txt')
#data = load_dataset("daily-min-temperatures.txt")
data = data[:, 1] # 1
plt.plot(data)
plt.show()

w = 150

from matrixprofile.algorithms.regimes import extract_regimes
from matrixprofile.algorithms.mpx import mpx
profile = mpx(np.array(data), w)
profile = extract_regimes(profile)
print(profile['regimes'])
plt.plot(profile['cac'])
plt.show()


oprofile, oindex = sc.matrixprofile.matrix_profile(data, w)
print(sc.matrixprofile.regimes(oprofile, oindex, w, -1, 5))




# # %%
# profile, index = sc.matrixprofile.matrix_profile(data, w)
# # plt.plot(profile)
# # plt.show()

# # %%
# cac, argmin = sc.matrixprofile.cac(profile, index, w)
# cac2 = sc.matrixprofile.cac_int(profile, index, w)
# argmin2 = sc.argmin(cac2)

# print(argmin, argmin2)


# plt.plot(cac)
# plt.show()
# plt.plot(cac2)
# plt.show()

# cac = sc.array(cac)

# print(cac.shape)
# print(cac2.shape)

# plt.plot(cac- cac2)
# plt.show()

# # %%

# # pos = sc.iota(index.shape, dtype = index.dtype)

# # smll = sc.minimum(pos, index)
# # large = sc.maximum(pos, index)

# # # sc.join([smll[1050:1150], pos[1050:1150], index[1050:1150]], 1)

# # smll = sc.sort(smll)
# # large = sc.sort(large)

# # si, sv = sc.sum_by_key(smll, sc.full(smll.shape, 1.0))
# # li, lv = sc.sum_by_key(large, sc.full(large.shape, -1.0))

# # mark1 = sc.zeros(index.shape)
# # mark1.assign(si, sv)
# # mark2 = sc.zeros(index.shape)
# # mark2.assign(li, lv)

# # mark = mark1 + mark2

# # cross_count = sc.cumsum(mark)


# # i = sc.iota(cross_count.shape)
# # l = cross_count.shape[0]
# # adj = 2.0 * i * (l-i) / l 
# # plt.plot(cross_count)
# # plt.plot(adj)
# # #plt.plot(adj / cross_count)
# # plt.show()


# # cross_count =cross_count /adj

# # cross_count = sc.minimum(cross_count, 1.)
# # fig, ax1 = plt.subplots() 
# # ax1.plot(data[5*w:-5*w], color="tab:green")
# # ax2=ax1.twinx()
# # ax2.plot(cross_count[5*w:-5*w], color="tab:red")
# # plt.show()

# # # %%

# %%
