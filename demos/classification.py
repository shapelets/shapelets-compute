#%%
import numpy as np
import matplotlib.pyplot as plt
import shapelets.compute as sc

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0].T
    x = data[:, 1:].T
    return sc.array(x), sc.array(y).astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
y_train[y_train == -1] = 0

x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
# %%


results = sc.clustering.kshape(x_train[:,:400], 2, max_iterations=1000)

# %%
plt.plot(results.centroids)
plt.show()
# %%
results.labels
# %%
y_train[:100]
# %%

sc.statistics.mean(sc.square(results.labels - y_train[:200]))
# %%

sc.sum(sc.round(sc.clip(results.labels * y_train[:200], 0, 1)))
# %%
sc.sum(sc.round(sc.clip(y_train[:200], 0, 1)))
# %%
sc.statistics.mean(y_train[:400] == results.labels)
# %%

def statistics(y_true, y_pred):
    equals = sc.statistics.mean(y_true == y_pred)
    indices_for_zero = 

