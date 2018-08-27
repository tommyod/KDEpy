#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the BaseKDE class.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numbers
import numpy as np
from KDEpy.kernel_funcs import _kernel_functions
from KDEpy.bw_selection import _bw_methods
from KDEpy.utils import autogrid


class BaseKDE(ABC):
    """
    Abstract Base Class for every kernel density estimator.
    
    This class is never instantiated, it merely defines some common methods
    which every subclass must implement. In summary, it facilitates:
        
        - The `_available_kernels` parameter
        - Correct handling of `kernel` and `bw` in __init__
        - Forces subclasses to implement `fit(data)`, converts `data` to 
          correct shape (obs, dims) and converts `weights` to correct shape
          (obs,)
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
        
        # Verify that the choice of kernel is valid, and set the function
        akernels = sorted(list(self._available_kernels.keys()))
        msg = f'Kernel must be a string or callable. Options: {akernels}'
        if isinstance(kernel, str):
            kernel = kernel.strip().lower()
            if kernel not in akernels:
                raise ValueError(msg)
            else:
                self.kernel = self._available_kernels[kernel]
        elif callable(kernel):
            self.kernel = kernel
        else:
            raise ValueError(msg)
        
        # The `bw` paramter may either be a positive number, a string, or 
        # array-like such that each point in the data has a uniue bw
        if (isinstance(bw, numbers.Number) and bw > 0):
            self.bw = bw
        elif isinstance(bw, str):
            amethods = sorted(list(self._bw_methods.keys()))
            if bw.lower() not in set(m.lower() for m in amethods):
                msg = f'Kernel not recognized. Options are: {amethods}'
                raise ValueError(msg)
            self.bw = self._bw_methods[bw]
        elif isinstance(bw, (np.ndarray, Sequence)):
            self.bw = bw
        else:
            raise ValueError(f'Bandwidth must be > 0, array-like or a string.')
            
        # Test quickly that the method has done what is was supposed to do
        assert callable(self.kernel)
        assert (isinstance(self.bw, (np.ndarray, Sequence, numbers.Number)) or 
                callable(self.bw))
            
    @abstractmethod
    def fit(self, data, weights=None):
        """
        Fit the kernel density estimator to the data.
        This method converts the data to shape (obs, dims) and the weights
        to (obs,).

        Parameters
        ----------
        data : array-like or Sequence
            May be array-like of shape (obs,), shape (obs, dims) or a
            Python Sequence, e.g. a list or tuple.
        weights : array-like, Sequence or None
            May be array-like of shape (obs,), shape (obs, dims), a 
            Python Sequence, e.g. a list or tuple, or None.
        """
        
        # -------------- Set up the data depending on input -------------------
        # In the end, the data should be an ndarray of shape (obs, dims)
        data = self._process_sequence(data)
            
        obs, dims = data.shape

        if not obs > 0:
            raise ValueError('Data must contain at least one data point.')
        assert dims > 0
        self.data = data
        
        # -------------- Set up the weights depending on input ----------------
        if weights is not None:
            self.weights = self._process_sequence(weights).ravel()
            self.weights = self.weights / np.sum(self.weights)
            if not obs == len(self.weights):
                raise ValueError('Number of data obs must match weights')
        else:
            self.weights = weights
                
        # Test quickly that the method has done what is was supposed to do
        assert len(self.data.shape) == 2
        if self.weights is not None:
            assert len(self.weights.shape) == 1
            assert self.data.shape[0] == len(self.weights)
    
    @abstractmethod
    def evaluate(self, grid_points=None, bw_to_scalar=True):
        """
        Evaluate the kernel density estimator on the grid points.
        
        Parameters
        ----------
        grid_points : integer, tuple or array-like
            If an integer, the number of equidistant grid point in every
            dimension. If a tuple, the number of grid points in each
            dimension. If array-like, grid points of shape (obs, dims).
        """
        if not hasattr(self, 'data'):
            raise ValueError('Must call fit before evaluating.')
            
        # -------------- Set up the bandwidth depending on inputs -------------
        if isinstance(self.bw, (np.ndarray, Sequence)):
            if bw_to_scalar:
                bw = max(self.bw)
            else:
                bw = self.bw
        elif callable(self.bw):
            bw = self.bw(self.data)
        else:
            bw = self.bw
        self.bw = bw
            
        # -------------- Set up the grid depending on input -------------------
        # If the grid None or an integer, use that in the autogrid method
        types_for_autogrid = (numbers.Integral, tuple)
        if grid_points is None or isinstance(grid_points, types_for_autogrid):
            self._user_supplied_grid = False
            bw_grid = self.kernel.practical_support(bw)
            grid_points = autogrid(self.data, bw_grid, grid_points)
            # Set it here, so as not to call kernel.practical_support(bw) again
            self._kernel_practical_support = bw_grid
        else:
            self._user_supplied_grid = True
            grid_points = self._process_sequence(grid_points)
                
        obs, dims = grid_points.shape
        if not obs > 0:
            raise ValueError('Grid must contain at least one data point.') 
        self.grid_points = grid_points
    
        # Test quickly that the method has done what is was supposed to do
        if bw_to_scalar:
            assert isinstance(self.bw, numbers.Number)
            assert self.bw > 0
        assert len(self.grid_points.shape) == 2
           
    @staticmethod
    def _process_sequence(sequence_array_like):
        """
        Process a sequence of data input to ndarray of shape (obs, dims).

        Parameters
        ----------
        sequence_array_like : Sequence or array-like
            The input data.

        Examples
        --------
        >>> res = BaseKDE._process_sequence([1, 2, 3]) 
        >>> (res == np.array([[1], [2], [3]])).all()
        True
        """
        # Must convert to float to avoid possible interger overflow
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
        return np.asarray_chkfinite(out, dtype=np.float)
        
    def _evalate_return_logic(self, evaluated, grid_points):
        """
        Return either evaluation points y, or tuple (x, y) based on inputs.
        """
        # Adding epsilon to output helps contour plotting functions
        evaluated = evaluated.ravel() + np.finfo(float).eps
        obs, dims = grid_points.shape
        if self._user_supplied_grid:
            return evaluated
        else:
            if dims == 1:
                return grid_points.ravel(), evaluated
            return grid_points, evaluated
  
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
