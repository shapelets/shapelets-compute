
#%%
import shapelets.compute as sc
import shapelets.generators as sg 
import numpy as np 
import matplotlib.pyplot as plt 
import matrixprofile as morg
#%matplotlib inline


# %%
spec = [
    sg.cc_increasing() + sg.white_noise(),
    [[sg.cc_normal() + sg.white_noise(), sg.cc_cyclic()], [.6, .4]],
    sg.cc_decreasing() + sg.white_noise()
]
r = sg.generate(spec, 100, start_level=100.0, repetitions=10)
s = sg.generate(spec, 100, start_level=100.0, repetitions=10)
q = sg.generate([.2*sg.cc_cyclic()], 100, start_level=100, repetitions=1)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 7))
axes[0].plot(r)
axes[1].plot(s)
axes[2].plot(q)
fig.tight_layout()
plt.show()

# %%
profile = sc.matrix_profile(q, q.shape[0], r)
# profile = sc.matrix_profile(r, 200)

# %%
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
axes[0].plot(r)
axes[1].plot(profile.profile)
fig.tight_layout()
plt.show()

# %%

# %%
import pandas as pd
steam_df = pd.read_csv("https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1")
steam_df.head()
# %%
x = sc.array(steam_df)
# %%
x
# %%

# %%
x[:10, 0]
# %%
plt.plot(x[:,2])
plt.show()
# %%
plt.plot(steam_df)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from IPython.display import HTML
import os
m = 640
fig, axs = plt.subplots(2)
plt.suptitle('Steamgen Dataset', fontsize='30')
axs[0].set_ylabel("Steam Flow", fontsize='20')
axs[0].plot(steam_df['steam flow'], alpha=0.5, linewidth=1)
axs[0].plot(steam_df['steam flow'].iloc[643:643+m])
axs[0].plot(steam_df['steam flow'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time", fontsize='20')
axs[1].set_ylabel("Steam Flow", fontsize='20')
axs[1].plot(steam_df['steam flow'].values[643:643+m], color='C1')
axs[1].plot(steam_df['steam flow'].values[8724:8724+m], color='C2')
plt.show()
# %%
df = pd.read_csv("https://zenodo.org/record/4276348/files/Time_Series_Chains_Kohls_data.csv?download=1")
df.head()
# %%
x = sc.array(df)
# %%
profile = sc.matrix_profile(x, 20)

# %%
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
axes[0].plot(x)
axes[1].plot(profile.profile)
fig.tight_layout()
plt.show()
# %%
from datetime import datetime
import pandas_datareader.data as wb

stocklist = ['AAPL','GOOG','FB','AMZN','COP']

start = datetime(2016,6,8)
end = datetime(2016,6,11)

p = wb.DataReader(stocklist, 'yahoo',start,end)