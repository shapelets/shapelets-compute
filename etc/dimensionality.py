# %% 
import shapelets.compute as sc 
import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/justo.ruiz/Development/shapelets/solo_comprobacion/modules/shapelets/data/Patient1.Mix 02.amc.txt')
data = sc.array(data[:, 1]).astype("float32") 
x = sc.arange(data.shape[0], dtype="float32")
plt.plot(data)
plt.show()

# %%
result = sc.dimensionality.visvalingam(x, data, 50)
plt.plot(result[:,0], result[:,1])
plt.show()
# %%
result = sc.dimensionality.visvalingam(x, data, 150)
plt.plot(result[:,0], result[:,1], color="tab:red")
plt.plot(x, data, color="tab:blue")
plt.show()
print(result.shape)
print(data.shape)

# %%
r = sc.dimensionality.pip(x, data, 18)
plt.plot(x, data, color="tab:blue")
plt.scatter(r[:,0], r[:,1], marker=">")
plt.show()
# %%
r = sc.dimensionality.paa(x, data, 10)
plt.plot(r[:,0], r[:,1])
plt.show()

# %%
