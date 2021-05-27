import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import shapelets.compute as sc
import numpy as np
from shapelets.data import load_dataset 

day_ahead = load_dataset('day_ahead_prices')    

gb = 24
day_ahead_by_day = sc.unpack(day_ahead, gb, 1, gb, 1)

svd_results = sc.svd(day_ahead_by_day)

fig = plt.figure(figsize=(18, 12))

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1)
ax0.bar(range(svd_results.s.size), np.array(svd_results.pct), label='weight')
ax0.plot(svd_results.acc_pct,  marker='o', label='accumulated', color='red')
ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax0.set_ylim([0, 1.05])
ax0.set_title('Factor Contribution')
ax0.axhline(y = .9, color = 'g', linestyle='--')
ax0.legend()
ax1 = plt.subplot2grid((2,3), (0,1), colspan=3)
ax1.plot(svd_results.u[:,0])
ax1.set_title('Dominant Hourly Profile')
ax2 = plt.subplot2grid((2,3), (1,0), colspan=4)
ax2.set_title('Dominant Yearly Profile')
ax2.plot(svd_results.vt[0,:].T)
plt.tight_layout()
plt.show()
