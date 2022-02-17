#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
# 7f316466-ba56-4680-b178-d9138aec6d16

import matplotlib.pyplot as plt
import shapelets_compute.compute as sc
from shapelets_compute.data import load_dataset

import warnings

warnings.filterwarnings("ignore")

day_ahead_prices = load_dataset('day_ahead_prices')
solar_forecast = load_dataset('solar_forecast')

fig, ax = plt.subplots(2, figsize=(18, 16), sharex=True)
ax[0].plot(day_ahead_prices)
ax[0].set_title('Day-ahead enery prices')
ax[0].set_ylabel("EUR/MW")
ax[1].plot(solar_forecast)
ax[1].set_title('Solar Production forecast')
ax[1].set_ylabel("MW")
ax[1].set_xlabel("Day of year (1h freq)")
plt.show()

# %%
gb = 24
day_ahead_prices_by_day = sc.unpack(day_ahead_prices, gb, 1, gb, 1)
solar_forecast_by_day = sc.unpack(solar_forecast, gb, 1, gb, 1)

fig, ax = plt.subplots(2, figsize=(18, 16), sharex=True)
img1 = ax[0].imshow(day_ahead_prices_by_day, cmap='viridis', aspect='auto')
ax[0].set_title('Day-ahead enery prices')
ax[0].set_ylabel("Hour slot")
fig.colorbar(img1, ax=ax[0])

img2 = ax[1].imshow(solar_forecast_by_day, cmap='viridis', aspect='auto')
ax[1].set_title('Solar Production forecast')
ax[1].set_ylabel("Hour slot")
ax[1].set_xlabel("Day of year")
fig.colorbar(img2, ax=ax[1])
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(24, 16), sharex=True)
img1 = ax[0, 0].imshow(sc.diff1(day_ahead_prices_by_day), cmap='viridis', aspect='auto')
ax[0, 0].set_title('Diff1 day-ahead prices')
ax[0, 0].set_ylabel("Hour slot")
fig.colorbar(img1, ax=ax[0, 0])

img2 = ax[0, 1].imshow(sc.diff2(day_ahead_prices_by_day), cmap='viridis', aspect='auto')
ax[0, 1].set_title('Diff2 day-ahead prices')
fig.colorbar(img2, ax=ax[0, 1])

img3 = ax[1, 0].imshow(sc.diff1(solar_forecast_by_day), cmap='viridis', aspect='auto')
ax[1, 0].set_title('Diff1 solar forecast')
ax[1, 0].set_ylabel("Hour slot")
ax[1, 0].set_xlabel("Day of year")
fig.colorbar(img3, ax=ax[1, 0])

img4 = ax[1, 1].imshow(sc.diff2(solar_forecast_by_day), cmap='viridis', aspect='auto')
ax[1, 1].set_title('Diff2 solar forecast')
ax[1, 1].set_xlabel("Day of year")
fig.colorbar(img4, ax=ax[1, 1])
plt.show()

# %%
filter = sc.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype="float32")

filter /= sc.sum(filter)

r = sc.convolve2(solar_forecast_by_day, filter, 'default')
plt.imshow(r, cmap='magma', aspect='auto')
plt.colorbar()
plt.show()

rr = sc.pack(r, r.size, 1, gb, 1, gb, 1)
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(rr)
plt.show()

points = 24 * 7

fig, ax = plt.subplots(2, 1, figsize=(18, 8))
ax[0].plot(rr[:points])
ax[1].plot(solar_forecast_by_day[:points])
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(18, 8))
ax[0].plot(sc.fft.spectral_derivative(rr[:points]))
ax[1].plot(sc.fft.spectral_derivative(solar_forecast_by_day[:points]))
plt.show()

# %%
dataidx = sc.iota(data.size, dtype=data.dtype)
reduced = sc.dimensionality.visvalingam(dataidx, data, 1000)
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(reduced[:, 0], reduced[:, 1])
plt.show()

# %%
gb = 24
hour_day = sc.unpack(data, gb, 1, gb, 1)
plt.imshow(hour_day, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# %%
r = sc.convolve1(hour_day, [1, -2, 1.], 'default')
plt.imshow(r, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()
# %%
plt.imshow(sc.diff2(hour_day), cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# %%
plt.imshow(sc.diff1(hour_day), cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# %%
svd = sc.svd(hour_day)
low_rank = svd.low_rank(7)
plt.imshow(low_rank, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

# reconstructed = sc.pack(low_rank, low_rank.size, 1, gb, 1, gb, 1)
# fig, ax = plt.subplots(figsize=(18, 8))
# ax.plot(reconstructed)
# plt.show()
# fig, ax = plt.subplots(figsize=(18, 8))
# ax.plot(data[0:reconstructed.shape[0]] - reconstructed)
# plt.show()

# %%
fig, ax = plt.subplots(figsize=(18, 8))
ax.bar(svd.pct)
ax.plot(svd.acc_pct)
plt.show()

# %%
plt.plot(svd.u[:, 0])
plt.plot(svd.u[:, 1])
plt.show()
plt.plot(svd.vt[0, :].T)
plt.plot(svd.vt[1, :].T)
plt.show()
