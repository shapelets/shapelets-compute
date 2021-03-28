# %%
import pandas
import shapelets.compute as sc 
import numpy as np
import matplotlib.pyplot as plt 
# %%
x = sc.linspace(-2*np.pi, 2*np.pi, 10000)
y1 = sc.sin(x) 
y2 = sc.cos(x) 
y3 = y1*y2
y4 = sc.tan(x)
y5 = sc.tanh(x)
y6 = x*x -x/2.0 + 3.0
y7 = sc.sigmoid(x)

# %%
yy = sc.join([y1, y2, y3, y4, y5, y6, y7], 1)
yynp = np.array(yy)
print (yy.shape)
print(yynp.shape)

# %%

stdyy = sc.std(yy, 0)
t1 = sc.statistics.covariance(yy, False) / sc.matmulTN(stdyy, stdyy)
t2 = np.corrcoef(yy, rowvar=False)
if not t1.same_as(t2):
    print(t1)
    print(sc.array(t2))

# %%
sc.statistics.covariance(yy, True).same_as(np.cov(yy, rowvar=False))
# %%
f = plt.figure(figsize=(19,15))
plt.matshow(t1, fignum=f.number)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()

# %%
sc.statistics.covariance(yy, True)
# %%
