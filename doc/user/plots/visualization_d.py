import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.data import load_dataset 

solar_forecast = load_dataset('solar_forecast')    

gb = 24
solar_forecast_by_day = sc.unpack(solar_forecast, gb, 1, gb, 1)

filter = sc.array([
    [0, 5, 0],
    [5, 1, 5],
    [0, 5, 0]
], dtype= "float32") 

filter /= sc.sum(filter)

r = sc.convolve2(solar_forecast_by_day, filter, 'default')
rr = sc.pack(r, r.size, 1, gb, 1, gb, 1)

fig, ax = plt.subplots(2, 1, figsize=(18, 10))
ax[0].plot(solar_forecast)
ax[0].set_title("Original Series")
ax[1].plot(rr)
ax[1].set_title("Smooth Series")
plt.show()

points = 24*7

fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(solar_forecast_by_day[:points], label='Original')
ax.plot(rr[:points], label='Smooth')
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
ax[0].plot(sc.fft.spectral_derivative(solar_forecast_by_day[100:175]))
ax[0].set_title("Original Series")
ax[1].plot(sc.fft.spectral_derivative(rr[100:175]))
ax[1].set_title("Smooth Series")
plt.show()