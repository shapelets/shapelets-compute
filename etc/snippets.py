#%%
import matrixprofile as mp
import numpy as np
from matplotlib import pyplot as plt
from matrixprofile.visualize import plot_snippets
# ignore matplotlib warnings
import warnings
warnings.filterwarnings("ignore")


#%%
ts = np.loadtxt('InternalBleeding.txt')

plt.figure(figsize=(20,6))
plt.plot(ts,'g')
plt.title('Arterial Blood Pressure')
plt.ylabel('Data')
plt.show()
# %%
snippet_size = 177
num_snippets = 2

# Discover Snippets
snippets = mp.discover.snippets(ts, snippet_size, num_snippets)

# Print Snippet Metadata
for i in range(len(snippets)):
    print('** Snippet-' + str(i+1) + ' **')
    print('Index:', snippets[i]['index'])
    print('Fraction:', snippets[i]['fraction'])
    print()

# Visualize Snippets
plot_snippets(snippets, ts)
plt.show()
# %%
# %%
snippet_size = 177
window_size = int(np.floor(snippet_size / 2))
num_snippets = 2
time_series_len = len(ts) # 6939
n = len(ts)
# %%
# 141
num_zeros = int(snippet_size * np.ceil(n / snippet_size) - n)
ts = np.append(ts, np.zeros(num_zeros))

# %%
indices = np.arange(0, len(ts) - snippet_size, snippet_size)
distances = []
# %%
for j, i in enumerate(indices):
    print(j,i,i + snippet_size - 1,window_size)
#    distance = mpdist_vector(ts, ts[i:(i + snippet_size - 1)], int(window_size))
#    distances.append(distance)
# %%
def mass_distance_matrix(t,q,w):
    subseq_num = len(q) - w + 1
    distances = []
    
    for i in range(subseq_num):
        distances.append(np.real(mp.algorithms.mass2(t, q[i:i + w])))
    
    return np.array(distances)
    
# %%
xx = mass_distance_matrix(ts, ts[5841:6017], window_size)
# %%
xx.shape
# %%
import shapelets.compute as sc
sc.mass
# %%
def sc_mass_distance_matrix(t,q,w):
    sq = sc.array(q)
    st = sc.array(t)
    qq = sc.join([sq[i:i+w] for i in range(len(q)-w+1)],1)
    return sc.mass(qq,st)

# %%
mdm = sc_mass_distance_matrix(ts, ts[5841:6017], window_size)
# %%
mdm.shape
# %%
xx.T[0:10,0]
# %%
mdm[0:10,0:10]
# %%
sc.amin(mdm, 0)
# %%
scmp = sc.matrix_profile(ts, window_size, ts[5841:6017])

# %%
scmp.profile
# %%
scmp.index
# %%
inv = sc.matrix_profile(ts[5841:6017], window_size, ts)
# %%
inv.profile[100:200]
# %%
plt.plot(inv.profile)
plt.show()
plt.plot(scmp.profile)
plt.show()
# %%
