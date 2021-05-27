import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import shapelets.compute as sc
from shapelets.data import load_dataset 

day_ahead = load_dataset('day_ahead_prices')    

gb = 24
day_ahead_by_day = sc.unpack(day_ahead, gb, 1, gb, 1)

svd_results = sc.svd(day_ahead_by_day)
lr = svd_results.low_rank(1)
reconstructed = sc.pack(lr, lr.size, 1, gb, 1, gb, 1)

fig, ax = plt.subplots(2, 1, figsize=(18, 10))
ax[0].plot(day_ahead)
ax[0].set_title("Original Series")
ax[1].plot(reconstructed)
ax[1].set_title("From 1st Factor")
plt.show()

points = 24*7

fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(day_ahead_by_day[:points], label='Original')
ax.plot(reconstructed[:points], label='From 1st Factor')
plt.legend()
plt.show()
