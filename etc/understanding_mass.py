#%%
import shapelets.compute as sc
import matplotlib.pyplot as plt 

sc.set_backend(sc.Backend.OpenCL)

t = sc.array([10, 10, 10, 11, 12, 11, 10, 10, 11, 12, 11, 14, 10, 10])
q = sc.array([10, 10, 12, 12])
# %%
distances = sc.mass(sc.tile(q, (2,30)), sc.tile(t, (150,500)))
distances.eval()
# %%
distances.shape
# %%
