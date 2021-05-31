#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
import numpy as np
import matplotlib.pyplot as plt
import shapelets.compute as sc
from shapelets.data import load_mat

data = load_mat('ItalianPowerDemand.mat')[0:15000, 2]
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(data)
plt.show()

# %%
s = sc.matrixprofile.snippets(data, 200, 2, 50)
for idx, sn in enumerate(s):
    print("Snippet " + repr(idx) + " -> " + repr(round(sn['fraction'] * 100.0)))

# %%
margin = sc.amin(data) * .9
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(18, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
ax2.plot(s[0]['snippet'])
ax1.plot(s[1]['snippet'])
ax0.plot(data)
for idx, sn in enumerate(s):
    ax0.plot(sn['neighbors'], (margin * np.ones_like(sn['neighbors'])) + idx, 'o', label=repr(idx), markersize=7,
             alpha=.1)
plt.show()
