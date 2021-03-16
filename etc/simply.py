#%%
import shapelets as sh
# %%
a = sh.compute.array([1], (10,10))
# %%
type(a)
# %%
b = sh.compute.array([123,123])
# %%
b
# %%

sh.compute.fft.fft([10,10,13.], norm="ortho")

# %%
sh.compute.fft.fftfreq(10)
# %%
import numpy as np
np.fft.fftfreq(10)
# %%
