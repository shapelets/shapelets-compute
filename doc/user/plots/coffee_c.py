import matplotlib.pyplot as plt
import shapelets.compute as sc 
from shapelets.data import load_dataset
all_data = load_dataset('coffee_train')
ds = all_data[:,1:].T
labels = all_data[:,0]
ks = sc.clustering.KShape(2, rnd_labels=False)
ks.fit(ds, labels)
ks.plot_centroids() 