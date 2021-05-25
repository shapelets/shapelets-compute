import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.data import load_dataset 

day_ahead_prices = load_dataset('day_ahead_prices')
solar_forecast = load_dataset('solar_forecast')

fig, ax = plt.subplots(2, figsize=(18, 16), sharex=True)
ax[0].plot(day_ahead_prices)
ax[0].set_title('Day-ahead enery prices')
ax[0].set_ylabel("EUR/MW")
ax[1].plot(solar_forecast)
ax[1].set_title('Solar Production forecast')
ax[1].set_ylabel("MW")
ax[1].set_xlabel("Hour of the year")    
plt.show()
