import matplotlib.pyplot as plt
import shapelets.compute as sc 
from shapelets.data import load_dataset
all_data = load_dataset('coffee_train')
ds = all_data[:,1:].T
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(16,8))
sc.distances.dendogram_all_ts(ds, 'single', 'sbd', ts_hspace=5)
plt.tight_layout()
