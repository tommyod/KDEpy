#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file compares the speed of various implementations.
"""
import functools
import itertools
import operator
import time
import numpy as np
import os

from KDEpy import FFTKDE, TreeKDE, NaiveKDE
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

here = os.path.abspath(os.path.dirname(__file__))
save_path = os.path.join(here, r'../docs/source/_static/img/')


def timed(n=5):
    
    
    def time_function(function):
        
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            times = []
            for run in range(n):
                start_time = time.perf_counter()
                y = function(*args, **kwargs)
                if time.perf_counter() - start_time > 2:
                    return None
                times.append(time.perf_counter() - start_time)
                
            return times
        
        return wrapped
        
        
    return time_function


@timed()
def KDE_KDEpyFFTKDE(data):
    return FFTKDE().fit(data)()


@timed()
def KDE_scipy(data):
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data) - 1, np.max(data) + 1, num=2**10)
    return kde(x)


@timed()
def KDE_statsmodels(data):
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(fft=True, gridsize=2**10) # Estimate the densities
    x, y = kde.support, kde.density
    return y


@timed()
def KDE_sklearn(data):
    
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian', rtol=1E-4)
    kde.fit(data.reshape(-1, 1))
    
    # score_samples returns the log of the probability density
    x = np.linspace(np.min(data) - 1, np.max(data) + 1, num=2**10)
    logprob = kde.score_samples(x.reshape(-1, 1))
    return np.exp(logprob)
    

data_sizes_orig = [10**n for n in range(1, 9)]
data_sizes_orig = list(itertools.accumulate((5, 2) * 7, operator.mul))
data_sizes_orig = np.logspace(1, 8, num=15)

plt.figure(figsize=(8, 4))
plt.title('Profiling KDE implementations')
for function, name in zip([KDE_KDEpyFFTKDE, KDE_scipy, KDE_statsmodels, KDE_sklearn],
                          ['KDEpy.FFTKDE', 'scipy', 'statsmodels', 'sklearn']):
    agg_times = []
    data_sizes = []
    for data_size in data_sizes_orig:
        data = np.random.randn(int(data_size)) * np.random.randint(1, 10)
        times = function(data)
        print(data_size, times)
        if not times is None:
            agg_times.append(np.percentile(times, q= [10, 50, 90]))
            data_sizes.append(data_size)
        else:
            break
        
        
        
    plt.loglog(data_sizes, [t[1] for t in agg_times], zorder=15, label=name)
    plt.fill_between(data_sizes, 
                     [t[0] for t in agg_times], 
                     [t[2] for t in agg_times], alpha=0.5, zorder=-15)

plt.legend(loc='best')
plt.xlabel('Number of data points $N$')
plt.ylabel('Evaluation time $t$')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, r'profiling.png'))
plt.show()

