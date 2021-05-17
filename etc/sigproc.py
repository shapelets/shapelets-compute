# %%
import shapelets.compute as sc
import matplotlib.pyplot as plt
n = 1024
L = 60
dx = L / n
x = sc.arange(-L/2, L/2, dx)

f = sc.cos(x) * sc.exp(-sc.power(x, 2) / 25.0)
df = -(sc.sin(x) * sc.exp(-sc.power(x,2)/25) + (2.0/25)*x*f)

plt.plot(df)
plt.plot(sc.fft.spectral_derivative(f, L), color="tab:red")
plt.show()

# %%
import shapelets.compute as sc
import numpy as np
s = sc.linspace([100,-200.+0j], [3j,0], num = 5, endpoint=True, axis = 0)
n = np.linspace([100,-200.+0j], [3j,0], num = 5, endpoint=True, axis = 0)
print(s.shape)
print(n.shape)
s.display()
sc.array(n).display()
s.same_as(n)
# %%
sc.logspace(1, 4, num=4).same_as(np.logspace(1, 4, num=4))
# %%
sc.geomspace(1, 256, 56, True, dtype="float32")
# %%
