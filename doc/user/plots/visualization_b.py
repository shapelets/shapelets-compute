import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.data import load_dataset 

day_ahead_prices = load_dataset('day_ahead_prices')
solar_forecast = load_dataset('solar_forecast')

gb = 24
day_ahead_prices_by_day = sc.unpack(day_ahead_prices, gb, 1, gb, 1)
solar_forecast_by_day = sc.unpack(solar_forecast, gb, 1, gb, 1)

fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
img1 = ax[0].imshow(day_ahead_prices_by_day, cmap='magma', aspect='auto')
ax[0].set_title('Day-ahead enery prices')
ax[0].set_ylabel("Hour slot")
ax[0].set_xlabel("Day of year")  
fig.colorbar(img1, ax=ax[0])

img2 = ax[1].imshow(solar_forecast_by_day, cmap='magma', aspect='auto')
ax[1].set_title('Solar Production forecast')
ax[1].set_ylabel("Hour slot")
ax[1].set_xlabel("Day of year")    
fig.colorbar(img2, ax=ax[1])
plt.show()
