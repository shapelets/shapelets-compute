import matplotlib.pyplot as plt 
import shapelets.compute as sc
from shapelets.data import load_dataset 

heartbeat = load_dataset('ecg_heartbeat_av')    
heartbeat_matrix = sc.unpack(heartbeat, 32, 1, 32, 1)


fig = plt.figure(figsize=(18, 10))
ax0 = plt.subplot2grid((2,3), (0,0), colspan=3)
ax0.plot(heartbeat)
ax1 = plt.subplot2grid((2,3), (1,0))
ax1.imshow(heartbeat_matrix, cmap='magma', aspect='auto')
ax1.set_title("Matrix 32")
ax2 = plt.subplot2grid((2,3), (1,1))
ax2.imshow(sc.diff1(heartbeat_matrix), cmap='magma', aspect='auto')
ax2.set_title("Diff1")
ax3 = plt.subplot2grid((2,3), (1,2))
ax3.imshow(sc.diff2(heartbeat_matrix), cmap='magma', aspect='auto')
ax3.set_title("Diff2")
plt.tight_layout()
plt.show()