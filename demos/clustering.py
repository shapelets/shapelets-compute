#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
import shapelets.compute as sc
from shapelets.data import load_dataset
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

sc.set_backend('cpu')
dtype = "float64"

# raw_train = np.loadtxt('/Users/justo.ruiz/Development/shapelets/solo_comprobacion/demos/raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv')
# train_data = sc.array(raw_train[:, 1:])
# train_classes = sc.array(raw_train[:,0])
# train_classes[train_classes == -1] = 0

# raw_test = np.loadtxt('/Users/justo.ruiz/Development/shapelets/solo_comprobacion/demos/raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv')
# test_data = sc.array(raw_test[:, 1:])
# test_classes = sc.array(raw_test[:,0])
# test_classes[test_classes == -1] = 0


# raw_train = np.loadtxt('/Users/justo.ruiz/Downloads/ECG200/ECG200_TRAIN.txt')
# train_data = sc.array(raw_train[:, 1:])
# train_classes = sc.array(raw_train[:,0]) 
# train_classes[train_classes == -1] = 0
# print(sc.unique(train_classes))

# raw_test = np.loadtxt('/Users/justo.ruiz/Downloads/ECG200/ECG200_TEST.txt')
# test_data = sc.array(raw_test[:, 1:])
# test_classes = sc.array(raw_test[:,0])
# test_classes[test_classes == -1] = 0
# print(sc.unique(test_classes))

raw_train = load_dataset("Coffee_TRAIN.txt", dtype)
train_classes = raw_train[:, 0].astype("uint32")
train_data = raw_train[:, 1:]

raw_test = load_dataset("Coffee_TEST.txt", dtype)
test_classes = raw_test[:, 0].astype("uint32")
test_data = raw_test[:, 1:]

# %%
ks = sc.clustering.KShape(2, rnd_labels=True)
ks.fit(train_data.T, train_classes)
ks.plot_centroids()
prediction = ks.predit(test_data.T)

print(metrics.accuracy_score(np.array(train_classes), np.array(ks.labels_)))
print(metrics.accuracy_score(np.array(test_classes), np.array(prediction)))
print(metrics.confusion_matrix(np.array(test_classes), np.array(prediction)))

# %%
from tslearn.clustering import KShape as TSK

tks = TSK(n_clusters=2).fit(np.array(train_data), np.array(train_classes))
pred = tks.predict(np.array(test_data))

print(metrics.accuracy_score(np.array(train_classes), tks.labels_))
print(metrics.accuracy_score(np.array(test_classes), pred))
print(metrics.confusion_matrix(np.array(test_classes), pred))

plt.plot(tks.cluster_centers_[0])
plt.plot(tks.cluster_centers_[1])
plt.show()
