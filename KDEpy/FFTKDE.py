#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import pytest
import numbers
import numpy as np
from KDEpy.BaseKDE import BaseKDE
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.binning import linear_binning
from scipy.signal import convolve


class FFTKDE(BaseKDE):
    """
    Class for FFT implementation of the KDE.
    
    See pages 182-192 of [1].
    
    References
    ----------
    [1] Wand, M. P., and M. C. Jones. Kernel Smoothing. London ; New York: 
        Chapman and Hall/CRC, 1995.

    The class for a naive implementation of the KDE.
    """
    
    def __init__(self, kernel='gaussian', bw=1, norm=2):
        """
        Initialize a naive KDE.
        """
        super().__init__(kernel, bw)
        self.norm = norm
    
    def fit(self, data, weights=None):
        """Fit the KDE to the data.
    
        Parameters
        ----------
        data
            The data points.
        weights
            The weights.
            
        Returns
        -------
        self
            Returns the instance.
            
        Examples
        --------
        >>> data = [1, 3, 4, 7]
        >>> kde = NaiveKDE().fit(data)
        """
        
        # Sets self.data
        super().fit(data)
        
        # If weights were passed
        if weights is not None:
            if not len(weights) == len(data):
                raise ValueError('Length of data and weights must match.')
            else:
                weights = self._process_sequence(weights)
                self.weights = np.asfarray(weights)
        else:
            self.weights = np.ones_like(self.data)
            
        self.weights = self.weights / np.sum(self.weights)
            
        return self
    
    def evaluate(self, grid_points=None):
        """Evaluate on the grid points.
        """
        
        # This method sets self.grid points and verifies it
        super().evaluate(grid_points)
        
        # Return the array converted to a float type
        grid_points = np.asfarray(self.grid_points)
        
        # Step 1 - Obtaining the grid counts
        print(' Step 1 - Obtaining the grid counts')
        
        # TODO - Ensure linear binning will work with 1 grid point only
        # TODO: Ensure that linear binning will work in higher dimensions
        # TODO: Get rid of ravel()
        # TODO: Linear binning must take an argument which is an equidit grid
        num_grid_points = len(grid_points)  # 12
        grid_points, data = linear_binning(self.data.ravel(), 
                                           num_points=num_grid_points, 
                                           weights=self.weights.ravel())

        # Step 2 - Computing kernel weights
        print(' Step 2 - Computing kernel weights')
        
        # The self.bw is the number of standard deviations, find the true bw
        real_bw = self.bw / np.sqrt(self.kernel.var)
        
        # Compute dx for the grid
        dx = ((max(self.data.ravel()) - min(self.data.ravel())) / 
              (num_grid_points - 1))
        
        # Compute L, the number of dx'es to move out from 0 in kernel
        L = min(np.floor(real_bw / dx), num_grid_points - 1)
        
        # Evaluate the kernel once
        kernel_eval_grid = np.linspace(-dx * L, dx * L, int(L * 2 + 1))
        assert dx * L < real_bw
        kernel_weights = self.kernel(kernel_eval_grid, bw=self.bw).ravel()
        
        # Step 3 - Performing the convolution
        print(' Step 3 - Performing the convolution')
        
        evaluated = convolve(data, kernel_weights, mode='same').reshape(-1, 1)
        
        return self._evalate_return_logic(evaluated, grid_points)

        # Create zeros on the grid points
        evaluated = np.zeros_like(grid_points)
        
        # For every data point, compute the kernel and add to the grid
        bw = self.bw
        if isinstance(bw, numbers.Number):
            bw = np.asfarray(np.ones_like(self.data) * bw)
        elif callable(bw):
            bw = np.asfarray(np.ones_like(self.data) * bw(self.data))

        for weight, data_point, bw in zip(self.weights, self.data, bw):
            evaluated += weight * self.kernel(grid_points - data_point, 
                                              bw=bw)
            
        return self._evalate_return_logic(evaluated, grid_points)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    # Basic example of the naive KDE
    # -----------------------------------------
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    kernel = 'epa'
    bw = 1.2
    
    plt.figure(figsize=(10, 4))
    plt.title('Basic example of the naive KDE')
    
    plt.subplot(1, 2, 1)
    plt.title('FFTKDE')
    kde = FFTKDE(kernel=kernel, bw=bw)
    kde.fit(data)
    x = np.linspace(min(data), max(data), num=2**6)
    for d in data:
        k = NaiveKDE(kernel=kernel, bw=bw).fit([d]).evaluate(x) / len(data)
        plt.plot(x, k, color='k', ls='--')
        
    y = kde.evaluate(x)
    plt.plot(x, y)
    plt.scatter(data, np.zeros_like(data))
    
    plt.subplot(1, 2, 2)
    plt.title('NaiveKDE')
    kde = NaiveKDE(kernel=kernel, bw=bw)
    kde.fit(data)
    x = np.linspace(min(data), max(data), num=2**6)
    for d in data:
        k = NaiveKDE(kernel=kernel, bw=bw).fit([d]).evaluate(x) / len(data)
        plt.plot(x, k, color='k', ls='--')
        
    y = kde.evaluate(x)
    plt.plot(x, y)
    plt.scatter(data, np.zeros_like(data))
    plt.show()
    
    