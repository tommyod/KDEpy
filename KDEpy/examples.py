#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage examples.
"""

# --------- Minimal working example ---------
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
#plt.plot(x, norm(loc=0, scale=1).pdf(x), label='True distribution')
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

##
    