#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numbers
import numpy as np
from KDEpy.kernel_funcs import _kernel_functions
from KDEpy.bw_selection import _bw_methods


class KDE(object):
    
    _available_kernels = _kernel_functions
    _bw_methods = _bw_methods
    
    def __init__(self, kernel='gaussian', bw=1):
        """
        Initialize a new Kernel Density Estimator.
        
        kernel: must be a function integrating to 1
        bw : a bandwidth, i.e. a positive float
        """
        try:
            self.kernel = type(self)._available_kernels[kernel]
        except KeyError:
            self.kernel = kernel
         
        # Validate the inputs
        valid_methods = type(self)._bw_methods
        err_msg = "Parameter bw must be a positive number, or in ({})".format(
                  ' '.join(repr(m) for m in valid_methods))
        if not isinstance(bw, numbers.Real):
            if bw not in valid_methods:
                raise ValueError(err_msg)
        else:
            if bw <= 0:
                raise ValueError(err_msg)
                
        self.bw = bw

    def fit(self, data, boundaries=None):
        """
        Fit the kernel density estimator to the data.
        Boundaries may be a tuple.
        """
        self._data = np.asarray_chkfinite(data)
        
        if not boundaries:
            boundaries = (-np.inf, np.inf)
        self.boundaries = boundaries
        
        if np.all(np.diff(self._data) >= 0):
            self._data_sorted = True
        else:
            self._data_sorted = False
            
        return self
         
    def _set_weights(self, weights):
        if weights is None:
            weights = np.ones_like(self._data)
        weights = weights / np.sum(weights)
        return weights
    
    def _bw_selection(self):
        if isinstance(self.bw, (int, float)):
            return self.bw
        
        return _bw_methods[self.bw](self._data)
        
    def evaluate_naive(self, grid_points, weights=None):
        """
        Naive evaluation. Used primarily for testing.
        grid_points : np.array, evaluation points
        weights : np.array of weights for the data points, must sum to unity
        
        """
        grid_points = grid_points.astype(float)
        # If no weights are passed, weight each data point as unity
        weights = self._set_weights(weights)
        
        # Create zeros on the grid points
        evaluated = np.zeros_like(grid_points)
        
        # For every data point, compute the kernel and add to the grid
        bw = self._bw_selection()
        for weight, data_point in zip(weights, self._data):
            evaluated += weight * self.kernel(grid_points - data_point, bw=bw)
        
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
        bw_scale = self.bw / abs(self.kernel.right_bw + self.kernel.left_bw)
        
        # Compute the bandwidth to the left and to the right of the center
        left_bw = self.kernel.left_bw * bw_scale
        right_bw = self.kernel.right_bw * bw_scale

        j = np.searchsorted(data_sorted, grid_point + right_bw, side='right')
        # i = np.searchsorted(data_sorted[:j], grid_point - left_bw, 
        # side='left')
        i = np.searchsorted(data_sorted, grid_point - left_bw, side='left')
        i = np.maximum(0, i)
        j = np.minimum(len_data, j)
        
        # Get subsets of data and weights
        data_subset = data_sorted[i:j]
        weights_subset = weights_sorted[i:j]

        # Compute the values
        values = grid_point - data_subset
        kernel_estimates = self.kernel(values, bw=self.bw)
        weighted_estimates = np.dot(kernel_estimates, weights_subset)
        return np.sum(weighted_estimates)
    
    def evaluate_sorted(self, grid_points, weights=None):
        """
        Evaluated by sorting and using binary search.
        
        """
        len_data = len(self._data)
        
        # if len_grid_points > 2 * len_data:
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
    
    def evaluate(self, *args, **kwargs):    
        return self.evaluate_naive(*args, **kwargs)

       
def main():
    """
    %load_ext line_profiler
    %lprun -f slow_functions.main slow_functions.main()
    
    %lprun -f KDE.evaluate_sorted main()

    """
    
    x = np.linspace(-5, 5)
    data = np.random.random(10)
    KDE(bw='silverman').fit(data).evaluate_naive(x)
    
    
if __name__ == '__main__':
    main()
    
    
    