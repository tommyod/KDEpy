#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the FFTKDE.
"""
import pytest
import numbers
import numpy as np
from KDEpy.BaseKDE import BaseKDE
from KDEpy.binning import linear_binning
from scipy.signal import convolve
from KDEpy.utils import cartesian


class FFTKDE(BaseKDE):
    """
    This class implements a FFT computation of a 1D kernel density estimate. 
    While this implementation is very fast, there are some limitations: (1) the 
    bandwidth must be constant and (2) the KDE must be evaluated on an 
    equidistant grid. The finer the grid, the smaller the error.
    
    The evaluation step is split into two phases. First the :math:`N` data 
    points are binned using a linear binning routine on an equidistant grid `x`
    with :math:`n` grid points. This runs in :math:`O(N)` time.
    Then the kernel is evaluated once on :math:`\leq n` points and the result 
    of the kernel evaluation and the binned data is convolved. Using the
    convolution theorem, this step runs in :math:`O(n \log n)` time.
    While :math:`N` may be millions, :math:`n` is typically 2**10. The total
    running time of the algorithm is :math:`O(N + n \log n)`. See references.
    
    The implementation is reminiscent of the one found in statsmodels. However,
    ulike the statsmodels implementation every kernel is available for FFT
    computation, weighted data is available for FFT computation, and no large
    temporary arrays are created.

    Parameters
    ----------
    kernel : str
        The kernel function. See cls._available_kernels.keys() for choices.
    bw : float or str
        Bandwidth or bandwidth selection method. If a float is passed, it
        is the standard deviation of the kernel. If a string it passed, it
        is the bandwidth selection method, see cls._bw_methods.keys() for
        choices.
        
    Examples
    --------
    >>> data = np.random.randn(2**10)
    >>> # Automatic bw selection using Improved Sheather Jones
    >>> x, y = FFTKDE(bw='ISJ').fit(data).evaluate()
    >>> # Explicit choice of kernel and bw (standard deviation of kernel)
    >>> x, y = FFTKDE(kernel='triweight', bw=0.5).fit(data).evaluate()
    >>> weights = data + 10
    >>> # Using a grid and weights for the data
    >>> y = FFTKDE(kernel='epa', bw=0.5).fit(data, weights).evaluate(x)
    >>> # If you supply your own grid, it must be equidistant (use linspace)
    >>> y = FFTKDE().fit(data)(np.linspace(-10, 10, num=2**12))
    
    References
    ----------
    - Wand, M. P., and M. C. Jones. Kernel Smoothing. 
      London ; New York: Chapman and Hall/CRC, 1995. Pages 182-192.
    - Statsmodels implementation, at 
      ``statsmodels.nonparametric.kde.KDEUnivariate``.
    """
    
    def __init__(self, kernel='gaussian', bw=1):
        super().__init__(kernel, bw)
    
    def fit(self, data, weights=None):
        """
        Fit the KDE to the data. This validates the data and stores it. 
        Computations are performed upon evaluation on a grid.
    
        Parameters
        ----------
        data: array-like
            The data points.
        weights: array-like
            One weight per data point. Must have same shape as the data.
            
        Returns
        -------
        self
            Returns the instance.
            
        Examples
        --------
        >>> data = [1, 3, 4, 7]
        >>> weights = [3, 4, 2, 1]
        >>> kde = FFTKDE().fit(data, weights=None)
        >>> kde = FFTKDE().fit(data, weights=weights)
        >>> x, y = kde()
        """
        
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
            self.weights = np.ones(self.data.shape[0], dtype=np.float)
            
        self.weights = self.weights / np.sum(self.weights)
            
        return self
    
    def evaluate(self, grid_points=None):
        """
        Evaluate on the equidistant grid points.
        
        Parameters
        ----------
        grid_points: array-like or None
            A 1D grid (mesh) to evaluate on. If None, a grid will be 
            automatically created.
            
        Returns
        -------
        y: array-like
            If a grid is supplied, `y` is returned. If no grid is supplied,
            a tuple (`x`, `y`) is returned.
            
        Examples
        --------
        >>> kde = FFTKDE().fit([1, 3, 4, 7])
        >>> # Two ways to evaluate, either with a grid or without
        >>> x, y = kde.evaluate()
        >>> # kde.evaluate() is equivalent to kde()
        >>> y = kde(grid_points=np.linspace(0, 10, num=2**10))
        """
        
        # This method sets self.grid points and verifies it
        super().evaluate(grid_points)
        
        #print(self.grid_points.shape)
        #print(self._user_supplied_grid)
        
        # Return the array converted to a float type
        grid_points = np.asfarray(self.grid_points)
        
        # Verify that the grid is equidistant
        #diffs = np.diff(grid_points)
        #if not np.allclose(np.ones_like(diffs) * diffs[0], diffs):
        #    raise ValueError('The grid must be equidistant, use linspace.')
        
        if callable(self.bw):
            bw = self.bw(self.data)
        elif isinstance(self.bw, numbers.Number) and self.bw > 0:
            bw = self.bw
        else:
            raise ValueError('The bw must be a callable or a number.')
            
        self.bw = bw
        
            # Compute the number of grid points for each dimension in the grid
#        grid_num = np.array(list(len(np.unique(grid_points[:, i])) for 
#                                 i in range(dims)))
#        
#        # Scale the data to the grid
#        min_grid = np.min(grid_points, axis=0)
#        max_grid = np.max(grid_points, axis=0)
#        num_intervals = (grid_num - 1)  # Number of intervals
#        dx = (max_grid - min_grid) / num_intervals
#        data = (data - min_grid) / dx
        
        # Step 1 - Obtaining the grid counts
        num_grid_points = len(grid_points)
        data = linear_binning(self.data, 
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
        grids = [np.linspace(-dx * L, dx * L, int(L * 2 + 1))] 
        kernel_eval_grid = cartesian(grids)
        kernel_weights = self.kernel(kernel_eval_grid, bw=self.bw).ravel()
        
        #print(kernel_weights)
        # Step 3 - Performing the convolution
        evaluated = convolve(data, kernel_weights, mode='same').reshape(-1, 1)
        
        return self._evalate_return_logic(evaluated, grid_points)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])

if False:
    from KDEpy.utils import cartesian, autogrid
    import itertools
    import functools
    import operator
    
    # Create data
    data_orig = np.array([[0.6, 0.8],
                          [1,   2],
                          [1.8, 1.2]])
    
    grid_points = 8
    x, y = FFTKDE().fit(data_orig)(grid_points)
    
    print(x.shape)
    print(y.shape)
                
    #data_orig = np.array([[1,   2], [1, 3]])
    #n = 10000
    #data_orig = np.concatenate((np.random.randn(n).reshape(-1, 1) , 
    #                       np.random.randn(n).reshape(-1, 1)), axis=1)
    #data_orig = np.random.randn(50).reshape(-1, 1)
    #weights = np.random.randn(data_orig.shape[0])**2 + 100
    #weights = weights / np.sum(weights)
    
    #num_points = 12
    #grid_points = cartesian([np.linspace(-7, 7, num=num_points), np.linspace(-7, 7, num=12)])

    #result = linbin_2dim(data_orig, grid_points, weights=None)

    
    import matplotlib.pyplot as plt
    
    plt.scatter(data_orig[:, 0], data_orig[:, 1])#, s=weights * 1000)
    #print(result)
    output = y.reshape(grid_points, grid_points)
    plt.scatter(x[:, 0], x[:, 1], s = y * 1000, zorder = -1)
    
    plt.xticks(np.unique(x[:, 0]))
    plt.yticks(np.unique(x[:, 1]))
    plt.grid(True)
    plt.show()