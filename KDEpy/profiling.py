#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file compares the speed of various other implementations to that of
FFTKDE. It includes profiling a 1D example, a 2D example and several higher
dimensions.
"""


def main():
    import functools
    import time
    import numpy as np
    import os
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from KDEpy import FFTKDE
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import KernelDensity
    from KDEpy.utils import cartesian
    
    here = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(here, r'../docs/source/_static/img/')
    
    def timed(n=20, max_time=1, use_data_size=True):
        """
        Return a timing function running n times.
        """
        
        def time_function(function):
            @functools.wraps(function)
            def wrapped(data_size, *args, **kwargs):
                times = []
                for run in range(n):
                    if use_data_size:
                        np.random.seed(run)
                        data = np.random.randn(int(data_size))
                    else:
                        data = data_size
                    start_time = time.perf_counter()
                    function(data, *args, **kwargs)
                    if time.perf_counter() - start_time > max_time:
                        return None
                    times.append(time.perf_counter() - start_time)
                    
                return times
            return wrapped
        return time_function
    
    # -------------------------------------------------------------------------
    # --------- Profiling the 1D implementations ------------------------------
    # -------------------------------------------------------------------------
    
    @timed()
    def KDE_KDEpyFFTKDE(data, kernel='gaussian'):
        return FFTKDE(kernel=kernel).fit(data)()
    
    @timed()
    def KDE_scipy(data, kernel='gaussian'):
        kde = gaussian_kde(data)
        x = np.linspace(np.min(data) - 1, np.max(data) + 1, num=2**10)
        return kde(x)
    
    @timed()
    def KDE_statsmodels(data, kernel='gaussian'):
        fft = True
        if kernel == 'epa':
            fft = False
        kde = sm.nonparametric.KDEUnivariate(data)
        kde.fit(fft=fft, gridsize=2**10) 
        return kde.density
    
    @timed()
    def KDE_sklearn(data, kernel='gaussian'):
        if kernel == 'epa':
            kernel = 'epanechnikov'
        
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=1.0, kernel=kernel, rtol=1E-4)
        kde.fit(data.reshape(-1, 1))
        
        # score_samples returns the log of the probability density
        x = np.linspace(np.min(data) - 1, np.max(data) + 1, num=2**10)
        logprob = kde.score_samples(x.reshape(-1, 1))
        return np.exp(logprob)
        
    # Do profiling vs. other implementations in one dimentions.
    # Set up data, create the figure, perform the computations and create plot.
    data_sizes_orig = np.logspace(1, 8, num=15)
    plt.figure(figsize=(8, 4))
    plt.title('Profiling KDE implementations.' +
              r'Gaussian kernel on $2^{10}$ grid points.')  # Epanechnikov
    algorithms = [KDE_KDEpyFFTKDE, KDE_scipy, KDE_statsmodels, KDE_sklearn]
    names = ['KDEpy.FFTKDE', 'scipy', 'statsmodels', 'sklearn']
    for function, name in zip(algorithms, names):
        agg_times = []
        data_sizes = []
        for data_size in data_sizes_orig:
            print(name, data_size)
            times = function(data_size, kernel='gaussian')
            
            if times is not None:
                agg_times.append(np.percentile(times, q=[25, 50, 75]))
                data_sizes.append(data_size)
            else:
                break
            
        plt.loglog(data_sizes, [t[1] for t in agg_times], 
                   zorder=15, label=name)
        plt.fill_between(data_sizes, 
                         [t[0] for t in agg_times], 
                         [t[2] for t in agg_times], alpha=0.5, zorder=-15)
    
    plt.legend(loc='best')
    plt.xlabel('Number of data points $N$')
    plt.ylabel('Evaluation time $t$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, r'profiling_1D_gauss.png'))
    plt.show()
    
    # -------------------------------------------------------------------------
    # --------- Profiling the 2D implementations ------------------------------
    # -------------------------------------------------------------------------
    
    @timed()
    def KDE_KDEpyFFTKDE(data, kernel='gaussian'):
        data = np.concatenate((data.reshape(-1, 1), 
                               data.reshape(-1, 1) * 0.5), axis=1)
        x, y = FFTKDE(kernel=kernel).fit(data)((64, 64))
        assert len(y) == 64 * 64
        return y
    
    @timed()
    def KDE_scipy(data, kernel='gaussian'):
        eps = np.random.randn(len(data)) / 100
        kde = gaussian_kde(np.vstack([data, (data + eps) * 0.5]))
        X, Y = np.mgrid[-7:7:64j, -7:7:64j]
        x = np.vstack([X.ravel(), Y.ravel()])
        y = kde(x)
        assert len(y) == 64 * 64
        return y
    
    @timed()
    def KDE_statsmodels(data, kernel='gaussian'):
        data = [data.reshape(-1, 1), data.reshape(-1, 1) * 0.5]
        kde = sm.nonparametric.KDEMultivariate(data, var_type='cc')
        grid = cartesian([np.linspace(-7, 7, num=64), 
                          np.linspace(-7, 7, num=64)])
        y = kde.pdf(grid)
        assert len(y) == 64 * 64
        return y
    
    @timed()
    def KDE_sklearn(data, kernel='gaussian'):
        if kernel == 'epa':
            kernel = 'epanechnikov'
        
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=1.0, kernel=kernel, rtol=1E-4)
        data = np.concatenate((data.reshape(-1, 1), data.reshape(-1, 1) * 0.5), 
                              axis=1)
        kde.fit(data)
        
        # score_samples returns the log of the probability density
        linspace = np.linspace(-7, 7, num=64)
        grid = cartesian([linspace, linspace])
        logprob = kde.score_samples(grid)
        y = np.exp(logprob)
        assert len(y) == 64 * 64
        return y
        
    data_sizes_orig = np.logspace(1, 6, num=11)
    plt.figure(figsize=(8, 4))
    plt.title(r'Profiling KDE implementations. ' + 
              r'Gaussian kernel on $64 \times 64$ grid points.')
    functions = [KDE_KDEpyFFTKDE, KDE_scipy, KDE_statsmodels, KDE_sklearn]
    names = ['KDEpy.FFTKDE', 'scipy', 'statsmodels', 'sklearn']
    for function, name in zip(functions, names):
        
        agg_times = []
        data_sizes = []
        for data_size in data_sizes_orig:
            times = function(data_size, kernel='gaussian')
            print(name, data_size)
            
            if times is not None:
                agg_times.append(np.percentile(times, q=[25, 50, 75]))
                data_sizes.append(data_size)
            else:
                break
            
        plt.loglog(data_sizes, [t[1] for t in agg_times], 
                   zorder=15, label=name)
        plt.fill_between(data_sizes, 
                         [t[0] for t in agg_times], 
                         [t[2] for t in agg_times], alpha=0.5, zorder=-15)
    
    plt.legend(loc='upper left')
    plt.xlabel('Number of data points $N$')
    plt.ylabel('Evaluation time $t$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, r'profiling_2D_gauss.png'))
    plt.show()
        
    # -------------------------------------------------------------------------
    # --------- Profiling the FFTKDE on higher dimenions ----------------------
    # -------------------------------------------------------------------------
    
    @timed(n=40, max_time=3, use_data_size=False)
    def KDE_KDEpyFFTKDE(data, grid_pts, kernel='epa'):
        x, y = FFTKDE(kernel=kernel).fit(data)(grid_pts)
        return y
    
    plt.figure(figsize=(8, 4))
    plt.title(r'Profiling FFTKDE over dimensions on $\sim 4096$ grid points.')
    
    for data_size in [2, 3, 4, 5]:
        agg_times = []
        dims_list = []
        for dims in range(1, 9):
            
            np.random.seed(dims)
            gen = (np.random.randn(10**data_size).reshape(-1, 1) 
                   for i in range(dims))
            data = np.concatenate(tuple(gen), axis=1)
            print(data_size, dims)
            grid_pts = (int(np.round(4096**(1 / dims))),) * dims
            times = KDE_KDEpyFFTKDE(data, grid_pts, kernel='epa')
            
            if times is not None:
                agg_times.append(np.percentile(times, q=[25, 50, 75]))
                dims_list.append(dims)
            else:
                break
            
        plt.semilogy(dims_list, [t[1] for t in agg_times], 
                     zorder=15, label=f'$N = 10^{data_size}$')
        plt.fill_between(dims_list, [t[0] for t in agg_times], 
                         [t[2] for t in agg_times], alpha=0.5, zorder=-15)  
         
    plt.xticks(list(range(1, 9)))
    plt.legend(loc='upper left')
    plt.xlabel('Dimension $d$')
    plt.ylabel('Evaluation time $t$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, r'profiling_ND.png'))
    plt.show()
    

if __name__ == '__main__':
    main()
