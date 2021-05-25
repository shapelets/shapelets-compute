import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.data import load_dataset 

day_ahead_prices = load_dataset('day_ahead_prices')
solar_forecast = load_dataset('solar_forecast')    

gb = 24
day_ahead_prices_by_day = sc.unpack(day_ahead_prices, gb, 1, gb, 1)
solar_forecast_by_day = sc.unpack(solar_forecast, gb, 1, gb, 1)

fig, ax = plt.subplots(2, 2, figsize=(24, 16), sharex=True)
img1 = ax[0, 0].imshow(sc.diff1(day_ahead_prices_by_day), cmap='magma', aspect='auto')
ax[0, 0].set_title('Diff1 day-ahead prices')
ax[0, 0].set_ylabel("Hour slot")
fig.colorbar(img1, ax=ax[0, 0])

img2 = ax[0, 1].imshow(sc.diff2(day_ahead_prices_by_day), cmap='magma', aspect='auto')
ax[0, 1].set_title('Diff2 day-ahead prices')
fig.colorbar(img2, ax=ax[0, 1])

img3 = ax[1, 0].imshow(sc.diff1(solar_forecast_by_day), cmap='magma', aspect='auto')
ax[1, 0].set_title('Diff1 solar forecast')
ax[1, 0].set_ylabel("Hour slot")
ax[1, 0].set_xlabel("Day of year") 
fig.colorbar(img3, ax=ax[1, 0])

img4 = ax[1, 1].imshow(sc.diff2(solar_forecast_by_day), cmap='magma', aspect='auto')
ax[1, 1].set_title('Diff2 solar forecast')
ax[1, 1].set_xlabel("Day of year")    
fig.colorbar(img4, ax=ax[1, 1])    
plt.show()
