#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
import math
import numpy as np
import matplotlib.pyplot as plt

from shapelets.data import load_mat, load_dataset


# %% 
def ending_phase(c, mag):
    angle = math.asin(c[-1] / mag)
    if c[-2] > c[-1]:
        angle = np.pi - angle
    return angle


def next_phase(c, mag):
    ph1 = ending_phase(c[:-1], mag)
    ph2 = ending_phase(c, mag)
    return 2 * ph2 - ph1


def createData(noiseScale=0.001, freqBase=5.0, nextFreq=6.0):
    t = np.linspace(0, 1, 4000)
    mag = 10.0
    freq = freqBase
    phase = 0.0
    c1 = mag * np.sin(2 * np.pi * freq * t + phase)

    freq = nextFreq
    phase = next_phase(c1, mag)
    c2 = mag * np.sin(2 * np.pi * freq * t + phase)

    data = np.concatenate((c1, c2))
    data = data + np.random.normal(0.0, noiseScale, len(data))
    return data


# %%
# Simulate frequency oscillations in a
# high voltage distribution network, 
# oscillating at a base frequency of 
# 50Hz, with a change 0.1% change in 
# the middle of the data.
# Simulation data has 20% of noise

wLen = 800
noiseScale = 0.20
freqBase = 50.0
nextFreq = freqBase * 1.01

data = createData(noiseScale, freqBase, nextFreq)
plt.plot(data)
plt.title("Simulation Data")
plt.show()

# %%
# Run matrix profile on the data, using a window
# of 800 points 
import shapelets.compute as sc

profile, index, _ = sc.matrixprofile.matrix_profile(data, wLen)

# Use segmentation algorithm to identify 
# hidden changes in behaviour
minPoint = sc.matrixprofile.segment(profile, index, wLen, 1)[0]

# Plot results and the location of the 
# frequency shift
fig, ax = plt.subplots(2, 1, figsize=(18, 8))
ax[0].plot(profile)
ax[1].plot(data)
ax[1].axvspan(minPoint, minPoint + wLen, facecolor="yellow",
              edgecolor='none', alpha=0.5)
plt.show()

# %%
medts = load_dataset('ecg-heartbeat-av.txt')
wLen = 150
profile, index, _ = sc.matrixprofile.matrix_profile(medts, wLen)
r = sc.matrixprofile.segment(profile, index, wLen)
if len(r) > 0:
    plt.plot(medts, '--g')
    plt.plot(r, medts[sc.array(r)], 'rD')
    plt.show()

# %%
wLen = 800
data = load_mat('ItalianPowerDemand.mat')[10000:15000, 2]
profile, index, _ = sc.matrixprofile.matrix_profile(data, wLen)
minPoint = sc.matrixprofile.segment(profile, index, wLen, 1)[0]
fig, ax = plt.subplots(4, 1, figsize=(18, 8))
ax[0].plot(profile)
ax[1].plot(sc.matrixprofile.cac(profile, index, wLen))
ax[2].plot(data)
ax[2].axvspan(minPoint, minPoint + wLen, facecolor="yellow", edgecolor='none', alpha=0.5)
ax[3].plot(data[minPoint - wLen:minPoint + 2 * wLen])
ax[3].axvspan(wLen, 2 * wLen, facecolor="yellow", edgecolor='none', alpha=0.5)
plt.show()
