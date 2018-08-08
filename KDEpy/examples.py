#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage examples.
"""

# For Travis CI
import matplotlib
matplotlib.use('Agg')

# --------- Minimal working example ---------
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.figure(figsize=(6, 3))
##############################
from KDEpy import NaiveKDE, TreeKDE, FFTKDE
np.random.seed(42)
data = norm(loc=0, scale=1).rvs(2**3)
x, y = TreeKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
plt.plot(x, y, label='KDE estimate')
##############################
plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig(r'../example.png')



# --------- Minimal working example ---------
plt.figure(figsize=(6, 3))
##############################

data = norm(loc=0, scale=1).rvs(2**6)
for bw in [0.1, 'silverman', 1.5]:
    x, y = FFTKDE(kernel='triweight', bw=bw).fit(data).evaluate()
    plt.plot(x, y, label=f'KDE estimate, bw={bw}')

##############################
# plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig(r'../example2.png')

# --------- Every function used ---------

plt.figure(figsize=(6, 3))

np.random.seed(42)
data = norm(loc=0, scale=1).rvs(2**3)

for kde in [NaiveKDE, TreeKDE, FFTKDE]:
    x, y = kde(kernel='gaussian', bw='silverman').fit(data).evaluate()
    plt.plot(x, y, label=kde.__name__ + ' estimate')

plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout() 


# --------- Cool plot ---------

plt.figure(figsize=(8*1.2, 3.75*1.2))
np.random.seed(42)


plt.subplot(2, 3, 1)
n = 15
plt.title('Automatic bandwidth\n robust w.r.t multimodality')
data = np.concatenate((np.random.randn(n), np.random.randn(n) + 10))
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')
x, y = FFTKDE(bw='ISJ').fit(data)()
plt.plot(x, y, label='FFTKDE')
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)


plt.subplot(2, 3, 2)
plt.title('9+ kernel functions')
for kernel in FFTKDE._available_kernels.keys():
    x, y = FFTKDE(kernel=kernel).fit([0])()
    plt.plot(x, y, label=kernel)
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)  


plt.subplot(2, 3, 3)
plt.title('Fast 2D computations\nusing binning and FFT')
n = 16
data1 = np.concatenate((np.random.randn(n).reshape(-1, 1), 
                       np.random.randn(n).reshape(-1, 1)), axis=1)
data2 = np.concatenate((np.random.randn(n).reshape(-1, 1) + 1, 
                       np.random.randn(n).reshape(-1, 1) + 4), axis=1)
data = np.concatenate((data1, data2))

grid_points = 2**7  # Grid points in each dimension
N = 8  # Number of contours
x, z = FFTKDE(bw=1).fit(data)((grid_points, grid_points))
x, y = np.unique(x[:, 0]), np.unique(x[:, 1])
print(x.shape, y.shape, z.shape)
z = z.reshape(grid_points, grid_points).T
plt.contour(x, y, z, N, linewidths=0.8, colors='k')
plt.contourf(x, y, z, N, cmap="PuBu")
plt.plot(data[:, 0], data[:, 1], 'ok', ms=2)
plt.yticks([]); plt.xticks([]);


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm
ax = plt.subplot(2, 3, 4, projection='3d')
plt.title('Kernels normalized in any\ndimension for $p\in\{1, 2, \infty\}$')
data = np.array([[0, 0]])
grid_points = 2**6  # Grid points in each dimension
x, z = FFTKDE(kernel='gaussian', bw=1, norm=2).fit(data)((grid_points, grid_points))
x, y = np.unique(x[:, 0]), np.unique(x[:, 1])
x, y = np.meshgrid(x, y)
z = z.reshape(grid_points, grid_points).T + 0.1
ls = LightSource(350, 45)
rgb = ls.shade(z, cmap=cm.PuBu, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=True)
ax.view_init(30, 65)
plt.yticks([]); plt.xticks([]);
ax.set_zticks([])


plt.subplot(2, 3, 5)
plt.title('Individual data points\nmay be weighted')
np.random.seed(123)
n = 5
data = np.random.randn(n)*2
weights = np.random.randn(n) ** 2 + 1
kernel = 'triweight'
x, y = TreeKDE(kernel=kernel).fit(data, weights)()
plt.plot(x, y)
plt.scatter(data, np.zeros_like(data), s = weights * 20, color='red')
for d, w in zip(data, weights):
    y = TreeKDE(kernel=kernel).fit([d], weights=[w])(x) *  w / np.sum(weights)
    plt.plot(x, y, '--k', zorder=-15, alpha=0.75)
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15) 


plt.subplot(2, 3, 6)
data = np.random.gamma(10, 100, size=(10**6))
st = time.perf_counter()
x, y = FFTKDE(kernel='gaussian', bw=10).fit(data)(2**10)
timed = (time.perf_counter() - st)*1000
plt.plot(x, y)
plt.title('One million points on\n1024 points' + f' grid in {int(round(timed,0))} ms')
data = np.random.choice(data, size=100,replace=False)
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data', s=3)
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)

plt.tight_layout()
plt.savefig(r'../example3.png')
plt.savefig(r'../example3.pdf')



