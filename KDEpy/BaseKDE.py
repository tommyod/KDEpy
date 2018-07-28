#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
import pytest
import numbers
import numpy as np
from KDEpy.kernel_funcs import _kernel_functions
from KDEpy.bw_selection import _bw_methods


class BaseKDE(ABC):
    """
    Abstract Base Class for every kernel density estimator.
    
    This class is never instantiated, it merely defines some common methods
    which never subclass must implement. In summary, it facilitates:
        
        - The `_available_kernels` parameter
        - Correct handling of `kernel` and `bw` in __init__
        - Forces subclasses to implement `fit(data)`, converts `data` to 
          correct shape (obs, dims)
        - Forces subclasses to implement `evaluate(grid_points)`, with handling
    """
    
    _available_kernels = _kernel_functions
    _bw_methods = _bw_methods
    
    @abstractmethod
    def __init__(self, kernel: str, bw: float):
        """Initialize the kernel density estimator.

        The return type must be duplicated in the docstring to comply
        with the NumPy docstring style.
    
        Parameters
        ----------
        kernel
            Kernel function, or string matching available options.
        bw
            The bandwidth, either a number, a string or an array-like.
        """
        
        # Verify that the choice of a kernel is valid, and set the function
        akernels = sorted(list(self._available_kernels.keys()))
        msg = f'Kernel must be a string or callable. Options: {akernels}'
        if isinstance(kernel, str):
            kernel = kernel.strip().lower()
            
            if kernel not in akernels:
                raise ValueError(msg)
            
            self.kernel = self._available_kernels[kernel]
        elif callable(kernel):
            self.kernel = kernel
        else:
            raise ValueError(msg)
        
        # bw may either be a positive number, a string, or array-like such that
        # each point in the data has a uniue bw
        if (isinstance(bw, numbers.Number) and bw > 0):
            self.bw = bw
        elif isinstance(bw, str):
            kernel = kernel.strip().lower()
            amethods = sorted(list(self._bw_methods.keys()))
            if bw not in amethods:
                msg = f'Kernel not recognized. Options are: {amethods}'
                raise ValueError(msg)
            self.bw = self._bw_methods[bw]
        elif isinstance(bw, (np.ndarray, Sequence)):
            self.bw = bw
        else:
            raise ValueError(f'Bandwidth must be > 0, array-like or a string.')
            
    @abstractmethod
    def fit(self, data):
        """
        Fit the kernel density estimator to the data.
        """
        
        # In the end, the data should be an ndarray of shape (obs, dims)
        data = self._process_sequence(data)
            
        assert len(data.shape) == 2
        obs, dims = data.shape

        if not obs > 0:
            raise ValueError('Data must contain at least one data point.')
        assert dims > 0
        self.data = np.asfarray(data)
    
    @abstractmethod
    def evaluate(self, grid_points=None):
        """
        Evaluate the kernel density estimator.
        
        grid_points: positive integer (number of points), or a grid Sequence 
                     or ndarray of shape (obs, dims)
        """
        if not hasattr(self, 'data'):
            raise ValueError('Must call fit before evaluating.')
        
        # If no information is supplied at all, call the autogrid method
        if grid_points is None:
            self._user_supplied_grid = False
            
            if isinstance(self.bw, (np.ndarray, Sequence)):
                bw = max(self.bw)
            elif callable(self.bw):
                bw = self.bw(self.data)
            else:
                bw = self.bw
            grid_points = self._autogrid(self.data, 
                                         self.kernel.practical_support(bw))
            
        # If a number is specified, interpret it as the number of grid points
        elif isinstance(grid_points, numbers.Number):
            if not (isinstance(grid_points, numbers.Integral) and 
                    grid_points > 0):
                raise ValueError('grid_points must be positive integer.')
            self._user_supplied_grid = False
            grid_points = self._autogrid(self.data, num_points=grid_points)
            
        else:
            self._user_supplied_grid = True
            grid_points = self._process_sequence(grid_points)
                
        obs, dims = grid_points.shape
        if not obs > 0:
            raise ValueError('Grid must contain at least one data point.') 
            
        self.grid_points = grid_points
        
        assert hasattr(self, '_user_supplied_grid')
        
    def _process_sequence(self, sequence_array_like):
        """
        Process a sequence of data input to ndarray of shape (obs, dims).
        """
        if isinstance(sequence_array_like, Sequence):
            out = np.asfarray(sequence_array_like).reshape(-1, 1)
        elif isinstance(sequence_array_like, np.ndarray):
            if len(sequence_array_like.shape) == 1:
                out = sequence_array_like.reshape(-1, 1)
            elif len(sequence_array_like.shape) == 2:
                out = sequence_array_like
            else:
                raise ValueError('Must be of shape (obs, dims)')
        else:
            raise TypeError('Must be of shape (obs, dims)')
            
        return np.asarray_chkfinite(np.asfarray(out))
        
    def _evalate_return_logic(self, evaluated, grid_points):
        """
        Return either evaluation points y, or tuple (x, y) based on inputs.
        """
        obs, dims = evaluated.shape
        if self._user_supplied_grid:
            if dims == 1:
                return evaluated.ravel()
            return evaluated 
        else:
            if dims == 1:
                return grid_points.ravel(), evaluated.ravel()
            return grid_points, evaluated 
        
    @staticmethod
    def _autogrid(data, kernel_support, num_points=1024, percentile=0.05):
        """
        Automatically select a grid if the user did not supply one.
        
        number of grid : should be a power of two
        percentile : is how far out we go out
        """
        obs, dims = data.shape
        minimums, maximums = data.min(axis=0), data.max(axis=0)
        ranges = maximums - minimums
        
        grid_points = np.empty(shape=(num_points // 2**(dims - 1), dims))

        generator = enumerate(zip(minimums, maximums, ranges))
        for i, (minimum, maximum, rang) in generator:
            outside_borders = max(percentile * rang, kernel_support)
            grid_points[:, i] = np.linspace(minimum - outside_borders,
                                            maximum + outside_borders,
                                            num=num_points // 2**(dims - 1))

        return grid_points
        
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])