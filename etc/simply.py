#%%
import shapelets.compute as sc
import numpy as np 
import pathlib
import os
import matplotlib.pyplot as plt
import cProfile
import time 
np.set_printoptions(precision=4, suppress=True)

# sc.set_backend(sc.Backend.OpenCL)

current_path = pathlib.Path(__file__).parent.absolute()
# ts = sc.array(np.loadtxt(os.path.join(current_path, "..", "modules", "test", "sampledata.txt")))
s = time.perf_counter()
ts = sc.array(np.loadtxt(os.path.join(current_path, "InternalBleeding.txt")))
e = time.perf_counter() - s 
print("Loading data took: " + str(e) )

# %%
# import matrixprofile as morg 
# xx = np.array(ts)
# s = morg.discover.snippets(xx, 400, 2)
# for i in range(len(s)):
#     print('** Snippet-' + str(i+1) + ' **')
#     print('Index:', s[i]['index'])
#     print('Fraction:', s[i]['fraction'])
#     print()

# cProfile.run('morg.discover.snippets(xx, 400, 2)')

# %%

snippets = sc.matrixprofile.snippets(ts, 400, 2)
cProfile.run('sc.matrixprofile.snippets(ts, 400, 2)')
# %%

fig, ax = plt.subplots(figsize=(18, 8))
margin= (sc.amin(ts) * .9).real

ax.plot(ts)
for idx, sn in enumerate(snippets):
    ax.plot(sn.indices, (margin * np.ones_like(sn.indices)) + idx/15., 'o', label=repr(sn), markersize=20, alpha=.1)

ax.legend()
plt.show()

# %%
