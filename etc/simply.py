#%%
<<<<<<< HEAD
import shapelets.compute as sc
import numpy as np 
import pathlib
import os
import matplotlib.pyplot as plt
import cProfile
np.set_printoptions(precision=4, suppress=True)

sc.set_backend(sc.Backend.CPU)

current_path = pathlib.Path(__file__).parent.absolute()
ts = sc.array(np.loadtxt(os.path.join(current_path, "..", "modules", "test", "sampledata.txt")))

# sc.matrixprofile.mpdist_vect(ts, q, 25).display()
# cProfile.run('sc.matrixprofile.mpdist_vect(ts, q, 25)')
# cProfile.run('sc.matrixprofile.mpdist_vect(ts, q, 25)')

# print(morg.discover.hierarchical_clusters(xx, 50, 3))

# print(morg.algorithms.mpdist_vector(xx, xx[0:49], 25).shape)
# cProfile.run('morg.algorithms.mpdist_vector(xx, xx[0:49], 25)')
# cProfile.run('morg.algorithms.mpdist_vector(xx, xx[0:49], 25)')

# %%
import matrixprofile as morg 
xx = np.array(ts)
s = morg.discover.snippets(xx, 50, 2)
for i in range(len(s)):
    print('** Snippet-' + str(i+1) + ' **')
    print('Index:', s[i]['index'])
    print('Fraction:', s[i]['fraction'])
    print()

cProfile.run('morg.discover.snippets(xx, 50, 2)')

# %%

snippets = sc.matrixprofile.snippets(ts, 60, 3, 10)
cProfile.run('sc.matrixprofile.snippets(ts, 50, 2)')
# %%

fig, ax = plt.subplots(figsize=(18, 8))

ax.plot(ts)
for idx, sn in enumerate(snippets):
    ax.plot(sn.indices, (-2 * np.ones_like(sn.indices)) + idx/15., 'o', label=repr(sn), markersize=4, alpha=.5)

ax.legend()
plt.show()
=======
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
>>>>>>> master
