import shapelets.compute as sc 
import matplotlib.pyplot as plt 
from shapelets.data import load_dataset
robot = sc.normalization.zscore(load_dataset('robot_dog'))
query = sc.normalization.zscore(load_dataset('carpet_query'))
r = sc.matrixprofile.mass(query, robot)
index, _ = sc.argmin(r)
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(range(index-50, index+150), robot[index-50:index+150], label='series')
ax.plot(range(index, index+100), query, label='query')
ax.axvline(x=index, linestyle='--', color = 'k', alpha=0.4)
plt.legend()
plt.show()