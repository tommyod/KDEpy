#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the FFTKDE.
"""
import pytest
import numbers
import numpy as np
from KDEpy.BaseKDE import BaseKDE
from KDEpy.binning import linbin_numpy
from scipy.signal import convolve


class FFTKDE(BaseKDE):
    """
    Class for FFT implementation of the KDE. While this implementation is fast,
    there are three limitations: (1) The data must be unimodal, (2) the 
    bandwidth must be constant and (3) the KDE must be evaluated on an
    equidistant grid.
    
    See pages 182-192 of [1].
    
    References
    ----------
    [1] Wand, M. P., and M. C. Jones. Kernel Smoothing. London ; New York: 
        Chapman and Hall/CRC, 1995.
    """
    
    def __init__(self, kernel='gaussian', bw=1):
        """
        Initialize a naive KDE.
        """
        super().__init__(kernel, bw)
    
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
        >>> kde = FFTKDE().fit(data)
        >>> x = np.linspace(1, 7)
        >>> y = kde.evaluate(x)
        """
        
        # Since the FFT is only used for 1D KDEs, we check that the user inputs
        # are 1D if they are NumPy ndarrays
        class_name = type(self).__name__
        if isinstance(data, np.ndarray):
            if not len(data.shape) == 1:
                msg = 'The data for {} must be 1D'.format(class_name)
                raise ValueError(msg)
                
        if isinstance(weights, np.ndarray):
            if not len(weights.shape) == 1:
                msg = 'The weights for {} must be 1D'.format(class_name)
                raise ValueError(msg)
                
        # ------------- END code specific for FFTKDE --------------------------
        
        # Sets self.data
        super().fit(data)
        
        # If weights were passed
        if weights is not None:
            if not len(weights) == len(data):
                raise ValueError('Length of data and weights must match.')
            else:
                weights = self._process_sequence(weights)
                self.weights = np.asfarray(weights, dtype=np.float)
        else:
            self.weights = np.ones_like(self.data, dtype=np.float)
            
        self.weights = self.weights / np.sum(self.weights)
            
        return self
    
    def evaluate(self, grid_points=None):
        """
        Evaluate on the grid points.
        """
        
        # This method sets self.grid points and verifies it
        super().evaluate(grid_points)
        
        # Return the array converted to a float type
        grid_points = np.asfarray(self.grid_points)
        
        # Verify that the grid is equidistant
        diffs = np.diff(grid_points)
        if not np.allclose(np.ones_like(diffs) * diffs[0], diffs):
            raise ValueError('The grid must be equidistant, use linspace.')
        
        if callable(self.bw):
            bw = self.bw(self.data)
        elif isinstance(self.bw, numbers.Number) and self.bw > 0:
            bw = self.bw
        else:
            raise ValueError('The bw must be a callable or a number.')
            
        self.bw = bw
        
        # Step 1 - Obtaining the grid counts
        # TODO: Speed up even more using Cython?
        num_grid_points = len(grid_points)
        data = linbin_numpy(self.data.ravel(), 
                            grid_points=grid_points, 
                            weights=self.weights.ravel())
        
        # Step 2 - Computing kernel weights
        # Compute dx for the grid
        num_grid_points = len(grid_points)
        dx = ((self.grid_points.max() - self.grid_points.min()) / 
              (num_grid_points - 1))
        
        # Find the real bandwidth, the support times the desired bw factor
        if self.kernel.finite_support:
            real_bw = self.kernel.support * self.bw
        else:
            # TODO: Make this more robust with threshold
            real_bw = self.kernel.practical_support(self.bw)
            
        # Compute L, the number of dx'es to move out from 0 in kernel
        L = min(np.floor(real_bw / dx), num_grid_points - 1)
        assert dx * L < real_bw
        
        # Evaluate the kernel once
        kernel_eval_grid = np.linspace(-dx * L, dx * L, int(L * 2 + 1))
        kernel_weights = self.kernel(kernel_eval_grid, bw=self.bw).ravel()
        
        # Step 3 - Performing the convolution
        evaluated = convolve(data, kernel_weights, mode='same').reshape(-1, 1)
        
        return self._evalate_return_logic(evaluated, grid_points)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    