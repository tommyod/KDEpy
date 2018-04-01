#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import pytest
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

    def fit(self, data, weights=None, boundaries=None):
        """
        Fit the kernel density estimator to the data.
        Boundaries may be a tuple.
        """
        self._data = np.asarray_chkfinite(data)

        # If no weights are passed, weight each data point as unity
        self.weights = self._set_weights(weights)
        
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
            weights = np.asfarray(np.ones_like(self._data))
        weights = weights / np.sum(weights)
        return weights
    
    def _bw_selection(self):
        if isinstance(self.bw, (int, float)):
            return self.bw
        
        return _bw_methods[self.bw](self._data)
        
    def evaluate_naive(self, grid_points):
        """
        Naive evaluation. Used primarily for testing.
        grid_points : np.array, evaluation points
        weights : np.array of weights for the data points, must sum to unity
        
        """
        # Return the array converted to a float type
        grid_points = np.asfarray(grid_points)
        
        # Create zeros on the grid points
        evaluated = np.zeros_like(grid_points)
        
        # For every data point, compute the kernel and add to the grid
        bw = self._bw_selection()
        for weight, data_point in zip(self.weights, self._data):
            evaluated += weight * self.kernel(grid_points - data_point, bw=bw)
        
        return evaluated 
    
    def _eval_sorted(self, data_sorted, weights_sorted, grid_point, len_data,
                     tolerance):
        """
        Evaluate the sorted weights.
        
        Use binary search to find data points close to the grid point,
        evaluate the kernel function on the data points, sum and return.
        Runs in O(log(`data`)) + O(`datapoints close to grid_point`).
        """
        
        # ---------------------------------------------------------------------
        # -------- Compute the support to the left and right of the kernel ----
        # ---------------------------------------------------------------------
        
        # If the kernel has finite support, find the support by scaling the 
        # variance. This is done when a call is made later on.
        if self.kernel.finite_support:
            left_support = self.kernel.support[0] 
            right_support = self.kernel.support[1] 
        else:
            # Compute the support up to a tolerance
            # This code assumes the kernel is symmetric about 0
            # TODO: Extend this code for non-symmetric kernels
            
            # Scale relative tolerance to the height of the kernel at 0
            tolerance = self.kernel(0, bw=self.bw) * tolerance
            # Sample the x values and the function to the left of 0
            x_samples = np.linspace(-self.kernel.var * 10, 0, num=2**10)
            sampled_func = self.kernel(x_samples, bw=self.bw) - tolerance
            # Use binary search to find when the function equals the tolerance
            i = np.searchsorted(sampled_func, 0, side='right') - 1
            left_support, right_support = x_samples[i], abs(x_samples[i])
            assert self.kernel(x_samples[i], bw=self.bw) <= tolerance
            
        # ---------------------------------------------------------------------
        # -------- Use binary search to only compute for points close by ------
        # ---------------------------------------------------------------------
        
        j = np.searchsorted(data_sorted, grid_point + right_support, 
                            side='right')
        # i = np.searchsorted(data_sorted[:j], grid_point - left_bw, 
        # side='left')
        i = np.searchsorted(data_sorted, grid_point + left_support, 
                            side='left')
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
    
    def evaluate_sorted(self, grid_points, tolerance=10e-6):
        """
        Evaluated by sorting and using binary search.
        
        """
        len_data = len(self._data)
        
        # if len_grid_points > 2 * len_data:
        #    return self.evaluate_naive(grid_points, weights = weights)
            
        # If no weights are passed, weight each data point as unity
        weights = self.weights
            
        # Sort the data and the weights
        indices = np.argsort(self._data)
        data_sorted = self._data[indices]
        weights_sorted = weights[indices]

        evaluated = np.zeros_like(grid_points)
        
        for i, grid_point in enumerate(grid_points):
            evaluated[i] = self._eval_sorted(data_sorted, weights_sorted, 
                                             grid_point, len_data, tolerance)

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
    y = KDE('box', bw=1).fit(data).evaluate_sorted(x)
    
    print(y)
    

if __name__ == '__main__':
    main()
    

if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    
    
    