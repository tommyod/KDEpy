#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np

def main():
    
    from KDEpy import FFTKDE, NaiveKDE
    from KDEpy.binning import linear_binning
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(123)
    dist = stats.lognorm(1, 1)
    plt.figure(figsize=(14, 6))
    
    kernel = 'triweight'
    
    N = 10**3
    data = dist.rvs(int(N))
    plt.scatter(data, np.zeros_like(data), marker='|')
    x, y = NaiveKDE(bw='silverman', kernel=kernel).fit(data)(2**10)
    plt.plot(x, y, label='FFTKDE')
    plt.plot(x, dist.pdf(x), label='True')
    
    # -----------------------------------------------------------------------
    # Adaptive
    alpha = 1.9
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
    print(np.min(kde(data)))
    print(stats.mstats.gmean(kde(data)))
    print(kde.bw, bw)
    #bw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])/100
    plt.scatter(data, kde_data)
    y = NaiveKDE(bw=bw, kernel=kernel).fit(data, weights=None)(x)
    plt.plot(x, kde.bw*((kde(x) + 0) / stats.mstats.gmean(kde_data))**-alpha, label='bw')
    plt.plot(x, y + np.ones_like(x)*0.00, label='Adaptive')
    plt.ylim([0, 0.7])
    
    plt.legend()
    plt.show()
    
    print('-'*32)
    
    # -----------------------------------------------------------------------
    # Mirror at bounds
    plt.figure(figsize=(14, 6))
    
    # Beta distribution, where x=1 is a hard lower limit
    dist = stats.beta(a=1.05, b=3, loc=0, scale=1)
    
    # Plot the normal KDE and the true density
    data = dist.rvs(10**2)
    plt.figure(figsize=(14, 6))
    kde = FFTKDE(bw='silverman', kernel='triweight')
    x, y = kde.fit(data)(2**10)
    plt.figure(figsize=(14, 6))
    plt.plot(x, dist.pdf(x), label='True')
    plt.plot(x, y, label='FFTKDE')
    plt.scatter(data, np.zeros_like(data), marker='|')
    print(np.min(data), np.max(data))
    
    data_transformed = np.log(data)
    plt.scatter(data_transformed, np.zeros_like(data_transformed), marker='|')
    kde = FFTKDE(bw='silverman', kernel='triweight')
    x, y = kde.fit(data_transformed)(2**10)
    plt.plot(x, y, label='FFTKDE - transformed')
    
    print(x)
    print(y)
    plt.plot(np.exp(x), 2*np.exp(y)*(1 + y) - 2)
    
    plt.ylim([0, 3])
    plt.xlim([-1, 4])
    
    plt.legend()
    plt.show()
    
    
    # -------------------------------------------------------------------------
    # Data on a circle
    # Beta distribution, where x=1 is a hard lower limit
    np.random.seed(123)
    
    dist1 = stats.norm(loc=0, scale=1)
    dist2 = stats.norm(loc=20, scale=1)
    dist3 = stats.norm(loc=40, scale=1)
    data = np.hstack([dist1.rvs(10**3), dist2.rvs(10**3), dist3.rvs(10**3)])


    plt.figure(figsize=(14, 6))
    x, y = FFTKDE(bw='silverman').fit(data)()
    plt.plot(x, (dist1.pdf(x) + dist2.pdf(x)+ dist3.pdf(x)) / 3, label='True distribution')
    plt.plot(x, y, label="FFTKDE with Silverman's rule")

    y = FFTKDE(bw='ISJ').fit(data)(x)
    plt.plot(x, y, label="FFTKDE with Improved Sheather Jones (ISJ)")

    
    plt.legend()
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()
