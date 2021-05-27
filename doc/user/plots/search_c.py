import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import shapelets.compute as sc 
from shapelets.data import load_dataset

robot = sc.normalization.zscore(load_dataset('robot_dog'))
query = sc.normalization.zscore(load_dataset('carpet_query'))

r = sc.matrixprofile.mass(query, robot)
index, min_dst = sc.argmin(r)
indices, values = sc.sort_index(r)

q = np.quantile(values, 0.005)
selected_indices = indices[values <= q]
mean = sc.mean(selected_indices)
std = sc.std(selected_indices) * 1.5

fig, ax = plt.subplots(2,1,figsize=(16, 8),gridspec_kw={'height_ratios': [5, 1]})
ax[0].plot(robot)
ax[0].axvspan(mean-std, mean+std, color='yellow', alpha = 0.4)
ax[0].axvline(x=index, linestyle='--', color = 'k', alpha=0.8)

sns.histplot(np.array(selected_indices), kde=True, ax=ax[1])
ax[1].set(xlim=(0, robot.shape[0]))

fig.tight_layout()
plt.show()

