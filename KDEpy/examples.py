#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file
"""

# For Travis CI
import matplotlib
matplotlib.use('Agg')

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from KDEpy import NaiveKDE, TreeKDE, FFTKDE
here = os.path.abspath(os.path.dirname(__file__))
save_path = os.path.join(here, r'../docs/source/_static/img/')


# -----------------------------------------------------------------------------
# ------ ADVERTISEMENT PLOT: Create the plot that is shown in the README ------
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 5.5))
np.random.seed(42)
FONTSIZE = 15


plt.subplot(2, 3, 1)
n = 15
plt.title('Automatic bandwidth,\nrobust w.r.t multimodality', fontsize=FONTSIZE)
data = np.concatenate((np.random.randn(n), np.random.randn(n) + 10))
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')
x, y = FFTKDE(bw='ISJ').fit(data)()
plt.plot(x, y, label='FFTKDE')
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)


plt.subplot(2, 3, 2)
plt.title('9+ kernel functions', fontsize=FONTSIZE)
for kernel in FFTKDE._available_kernels.keys():
    x, y = FFTKDE(kernel=kernel).fit([0])()
    plt.plot(x, y, label=kernel)
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)


plt.subplot(2, 3, 3)
plt.title('Fast 2D computations\nusing binning and FFT', fontsize=FONTSIZE)
n = 16
gen_random = lambda n: np.random.randn(n).reshape(-1, 1)
data1 = np.concatenate((gen_random(n), gen_random(n)), axis=1)
data2 = np.concatenate((gen_random(n) + 1, gen_random(n) + 4), axis=1)
data = np.concatenate((data1, data2))


grid_points = 2**7  # Grid points in each dimension
N = 8  # Number of contours
x, z = FFTKDE(bw=1).fit(data)((grid_points, grid_points))
x, y = np.unique(x[:, 0]), np.unique(x[:, 1])
z = z.reshape(grid_points, grid_points).T
plt.contour(x, y, z, N, linewidths=0.8, colors='k')
plt.contourf(x, y, z, N, cmap="PuBu")
plt.plot(data[:, 0], data[:, 1], 'ok', ms=2)
plt.yticks([]); plt.xticks([]);


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm
ax = plt.subplot(2, 3, 4, projection='3d')
plt.title('Kernels normalized in any\ndimension for $p\in\{1, 2, \infty\}$', fontsize=FONTSIZE)
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
plt.title('Individual data points\nmay be weighted', fontsize=FONTSIZE)
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
x, y = FFTKDE(kernel='gaussian', bw=100).fit(data)(2**10)
timed = (time.perf_counter() - st)*1000
plt.plot(x, y)
plt.title('One million observations on\n1024 grid' + f' points in {int(round(timed,0))} ms', fontsize=FONTSIZE)
data = np.random.choice(data, size=100,replace=False)
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data', s=3)
plt.yticks([]); plt.xticks([]);
plt.grid(True, ls='--', zorder=-15)


plt.tight_layout()
plt.savefig(os.path.join(save_path, r'showcase.png'))

# -----------------------------------------------------------------------------
# ------ MINIMAL WORKING EXAMPLE: Showing a simle way to create a plot --------
# -----------------------------------------------------------------------------

plt.figure(figsize=(6, 3))
##############################
np.random.seed(42)
data = norm(loc=0, scale=1).rvs(2**3)
x, y = TreeKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
plt.plot(x, y, label='KDE estimate')
##############################
plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(save_path, r'mwe.png'))


# -----------------------------------------------------------------------------
# ------ COMPARING BANDWIDTHS: Different bandwidths on the same data set ------
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 3))
##############################

data = norm(loc=0, scale=1).rvs(2**6)
for bw in [0.1, 'silverman', 1.5]:
    x, y = FFTKDE(kernel='triweight', bw=bw).fit(data).evaluate()
    plt.plot(x, y, label=f'KDE estimate, bw={bw}')

##############################
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout()
# plt.savefig(os.path.join(save_path, r'example2.png'))

# -----------------------------------------------------------------------------
# ------ EVERY ESTIMATOR: Comparing the different algorithms ------------------
# -----------------------------------------------------------------------------

plt.figure(figsize=(6, 3))

np.random.seed(42)
data = norm(loc=0, scale=1).rvs(2**3)

for kde in [NaiveKDE, TreeKDE, FFTKDE]:
    x, y = kde(kernel='gaussian', bw='silverman').fit(data).evaluate()
    plt.plot(x, y + np.random.randn()/100, label=kde.__name__ + ' estimate')

plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
plt.scatter(data, np.zeros_like(data), marker='|', color='red', label='Data')

plt.legend(loc='best')
plt.tight_layout()
