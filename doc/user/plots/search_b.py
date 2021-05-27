import shapelets.compute as sc 
import matplotlib.pyplot as plt 
import numpy as np
from shapelets.data import load_dataset
robot = sc.normalization.zscore(load_dataset('robot_dog'))
query = sc.normalization.zscore(load_dataset('carpet_query'))

r = sc.matrixprofile.mass(query, robot)
indices, values = sc.sort_index(r)

fig, ax = plt.subplots(figsize=(16, 8))
for i in np.array(indices[:10]):
    ax.plot(robot[i:i+100], color='r', alpha=0.1)
ax.plot(query)
plt.show()
