#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np


import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def main():
    
    from KDEpy import FFTKDE, NaiveKDE
    from KDEpy.binning import linear_binning
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(123)
    dist = stats.lognorm(0.8, 1)
    plt.figure(figsize=(14, 6))
    
    kernel = 'triweight'
    
    N = 10**2
    data = dist.rvs(int(N))
    plt.scatter(data, np.zeros_like(data), marker='|')
    x, y = NaiveKDE(bw='silverman', kernel=kernel).fit(data)(2**10)
    plt.plot(x, y, label='FFTKDE')
    plt.plot(x, dist.pdf(x), label='True')
    
    # -----------------------------------------------------------------------
    # Adaptive
    alpha = 2
    bw = 'silverman'
    kde = NaiveKDE(kernel='epa', bw=bw)
    kde.fit(data)(x)

    #y = NaiveKDE(bw=kde.bw*lambda_i).fit(x, weights=binned_data*lambda_i)(x)
    #plt.plot(x, y + np.ones_like(x)*0.00, label='Adaptive')
    
    # The FFTKDE grid may be wrong, but the true density cannot be
    # smaller than (1/N) K(0) at a given point
    min_kde = (1 / int(N)) * kde.kernel(0)
    kde_data = np.maximum(min_kde, kde(data))
    kde_data = kde(data)
    bw = kde.bw*((kde_data) / stats.mstats.gmean(kde_data))**-alpha
    #print(np.min(kde(data)))
    #print(stats.mstats.gmean(kde(data)))
    #print(kde.bw, bw)
    #bw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])/100
    plt.scatter(data, kde_data)
    y = NaiveKDE(bw=bw, kernel=kernel).fit(data, weights=None)(x)
    plt.plot(x, kde.bw*((kde(x) + 0) / stats.mstats.gmean(kde_data))**-alpha, label='bw')
    plt.plot(x, y + np.ones_like(x)*0.00, label='Adaptive')
    plt.ylim([0, 0.7])
    
    plt.legend()
    plt.show()
    
    
def test():
    
    from KDEpy import FFTKDE, NaiveKDE
    from KDEpy.binning import linear_binning
    import matplotlib.pyplot as plt
    from scipy import stats
    
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from KDEpy import FFTKDE  # Fastest 1D algorithm
    
    np.random.seed(123)  # Seed generator for reproducible results
    
    distribution = norm()  # Create normal distribution
    data = distribution.rvs(32)  # Draw random samples

    
    np.random.seed(123)
    data = distribution.rvs(32)
    
    # Use a box function with the FFTKDE to obtain a density estimate
    x, y = FFTKDE(kernel='box', bw=0.7).fit(data).evaluate()
    
#    plt.plot(x, y, zorder=10, color='#ff7f0e', label='KDE with box kernel')
#    plt.scatter(data, np.zeros_like(data), marker='|', c='r', label='Data', zorder=9)
#    plt.hist(data, density=True, label='Histogram', edgecolor='#1f77b4', color='w')
#    plt.plot(x, distribution.pdf(x), label='True pdf', c='r', ls='--')
#    plt.legend(loc='best');
    
    
if __name__ == "__main__":
    test()
