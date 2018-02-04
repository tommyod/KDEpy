#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""

import numpy as np
import time
import collections.abc 
from kernel_funcs import _kernel_functions

class KDE(object):
    
    _available_kernels = _kernel_functions
    
    def __init__(self, kernel='gaussian', bw=None):
        """
        Initialize a new Kernel Density Estimator.
        
        kernel: must be a function integrating to 1
        bw : a bandwidth, i.e. a positive float
        """
        try:
            self.kernel = type(self)._available_kernels[kernel]
        except:
            self.kernel = kernel
            
        self.bw = bw

    
    def fit(self, data):
        """
        Fit the kernel density estimator to the data.
        """
        self._data = np.asarray_chkfinite(data)
        
        if np.all(np.diff(self._data) >= 0):
            self._data_sorted = True
        else:
            self._data_sorted = False
         

    def _set_weights(self, weights):
        if weights is None:
            weights = np.ones_like(self._data)
        weights = weights / np.sum(weights)
        return weights

    def evaluate_naive(self, grid_points, weights = None):
        """
        Naive evaluation. Used primarily for testing.
        grid_points : np.array, evaluation points
        weights : np.array of weights for the data points, must sum to unity
        
        """
        # If no weights are passed, weight each data point as unity
        weights = self._set_weights(weights)
        
        # Create zeros on the grid points
        evaluated = np.zeros_like(grid_points)
        
        # For every data point, compute the kernel and add to the grid
        for weight, data_point in zip(weights, self._data):
            evaluated += weight * self.kernel(grid_points - data_point, 
                                              bw=self.bw)
        
        return evaluated 
    
    def _eval_sorted(self, data_sorted, weights_sorted, grid_point, len_data):
        """
        Evaluate the sorted weights.
        
        Use binary search to find data points close to the grid point,
        evaluate the kernel function on the data points, sum and return.
        Runs in O(log(`data`)) + O(`datapoints close to grid_point`).
        """
        
        # The relationship between the desired bandwidth and the original
        # bandwidth of the kernel function
        bw_scale = self.bw / abs(self.kernel.right_bw  + self.kernel.left_bw)
        
        # Compute the bandwidth to the left and to the right of the center
        left_bw = self.kernel.left_bw * bw_scale
        right_bw = self.kernel.right_bw * bw_scale

        j = np.searchsorted(data_sorted, grid_point + right_bw, side='right')
        #i = np.searchsorted(data_sorted[:j], grid_point - left_bw, side='left')
        i = np.searchsorted(data_sorted, grid_point - left_bw, side='left')
        i = np.maximum(0, i)
        j = np.minimum(len_data, j)
        
        # Get subsets of data and weights
        data_subset = data_sorted[i:j]
        weights_subset = weights_sorted[i:j]

        # Compute the values
        values = grid_point - data_subset
        kernel_estimates = self.kernel(values, bw = self.bw)
        weighted_estimates = np.dot(kernel_estimates, weights_subset)
        return np.sum(weighted_estimates)
    

    def evaluate_sorted(self, grid_points, weights = None):
        """
        Evaluated by sorting and using binary search.
        
        """
        len_data = len(self._data)
        
        #if len_grid_points > 2 * len_data:
        #    return self.evaluate_naive(grid_points, weights = weights)
            
        # If no weights are passed, weight each data point as unity
        weights = self._set_weights(weights)
            
        # Sort the data and the weights
        indices = np.argsort(self._data)
        data_sorted = self._data[indices]
        weights_sorted = weights[indices]
        
        
        
        evaluated = np.zeros_like(grid_points)
        
        for i, grid_point in enumerate(grid_points):
            evaluated[i] = self._eval_sorted(data_sorted, weights_sorted, 
                     grid_point, len_data)

        # Normalize, but do not divide by zero
        return evaluated
    
    
    
def main():
    """
    %load_ext line_profiler
    %lprun -f slow_functions.main slow_functions.main()

    """
    import matplotlib.pyplot as plt
    np.random.seed(123)
    n = 2**15
    print(n)
    data = np.concatenate([np.random.randn(n), np.random.randn(n) + 5])*1
    
    
    data = np.array([0, 0.1, 0.2, 0.3, 0.4, 2, 3, 4])
    
    kde = KDE(kernel = 'gaussian', bw = 0.6)
    kde.fit(data)
    
    x = np.linspace(np.min(data)-1, np.max(data)+1, num = 2**10)
    #weights = np.array([1, 2, 3, 4, 3, 2, 1, 0])
    weights = None #np.arange(len(data))  + 1
    #x = np.linspace(-2, 2+5, num = 5+5)
    st = time.perf_counter()
    y = kde.evaluate_sorted(x, weights = weights)
    speed = time.perf_counter() - st
    print('Computation in', speed)
    

    
    st = time.perf_counter()
    y_naive = kde.evaluate_naive(x, weights = weights)
    speed_naive = time.perf_counter() - st
    print('Naive computation in', speed_naive)
    print('Speedup:', round(speed_naive / speed, 3))
    
    print('Data / grid : ', round(len(data)/len(x), 3))
    print('log(data) / grid : ', round(np.log(len(data))/len(x),4))
    
    #data_sampled = np.random.choice(data, n, replace = False)
    plt.scatter(data, np.zeros_like(data), 
                color = 'red', marker = 'x', label = 'data')
    plt.plot(x, y_naive, label = 'naive')
    plt.plot(x, y, label = 'sorted')
    plt.legend(loc = 'best')
    
    plt.ylim([0, max(np.max(y), np.max(y_naive))*1.05])
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    main()
    
    
    