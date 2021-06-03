#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the BaseKDE class.
"""
import numbers
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from KDEpy.kernel_funcs import _kernel_functions
from KDEpy.bw_selection import _bw_methods, cross_val
from KDEpy.utils import autogrid


class BaseKDE(ABC):
    """
    Abstract Base Class for every Kernel Density Estimator.

    This class is never instantiated, it merely defines some common methods
    which every subclass must implement. In summary, it facilitates:

        - The `_available_kernels` and `_bw_methods` parameter
        - Correct handling of `kernel` and `bw` in __init__
        - Forces subclasses to implement `fit(data)`, converts `data` to
          correct shape (obs, dims) and converts `weights` to correct shape
          (obs,)
        - Forces subclasses to implement `evaluate(grid_points)`, with handling
    """

    _available_kernels = _kernel_functions
    _bw_methods = _bw_methods

    @abstractmethod
    def __init__(self, kernel: str, bw: float, norm: float):
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
        msg = "Kernel must be a string or callable. Opts: {}".format(akernels)
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

        # CV method must be added here since it depends on self
        _bw_methods["CV"] = self.cross_val

        # The `bw` paramter may either be a positive number, a string, or
        # array-like such that each point in the data has a uniue bw
        if isinstance(bw, numbers.Number) and bw > 0:
            self.bw_method = bw
        elif isinstance(bw, str):
            amethods = sorted(list(self._bw_methods.keys()))
            if bw.lower() not in set(m.lower() for m in amethods):
                msg = "bw not recognized. Options are: {}".format(amethods)
                raise ValueError(msg)
            self.bw_method = self._bw_methods[bw]
        elif isinstance(bw, (np.ndarray, Sequence)):
            self.bw_method = bw
        else:
            raise ValueError("Bandwidth must be > 0, array-like or a string.")

        self.norm = norm

        # Test quickly that the method has done what is was supposed to do
        assert callable(self.kernel)
        assert isinstance(self.bw_method, (np.ndarray, Sequence, numbers.Number)) or callable(self.bw_method)
        assert isinstance(self.norm, numbers.Number) and self.norm > 0

    @abstractmethod
    def fit(self, data, weights=None, **kwargs):
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
        **kwargs:
            List of arguments to be passed to bandwidth optimization method.
        """

        # -------------- Set up the data depending on input -------------------
        # In the end, the data should be an ndarray of shape (obs, dims)
        data = self._process_sequence(data)

        obs, dims = data.shape

        if not obs > 0:
            raise ValueError("Data must contain at least one data point.")
        assert dims > 0
        self.data = data

        # -------------- Set up the weights depending on input ----------------
        if weights is not None:
            self.weights = self._process_sequence(weights).ravel()
            self.weights = self.weights / np.sum(self.weights)
            if not obs == len(self.weights):
                raise ValueError("Number of data obs must match weights")
        else:
            self.weights = weights

        # TODO: Move bandwidth selection from evaluate to fit

        # Test quickly that the method has done what is was supposed to do
        assert len(self.data.shape) == 2
        if self.weights is not None:
            assert len(self.weights.shape) == 1
            assert self.data.shape[0] == len(self.weights)

        if isinstance(self.bw_method, (np.ndarray, Sequence)):
            self.bw = self.bw_method
        elif callable(self.bw_method):
            self.bw = self.bw_method(self.data, self.weights, **kwargs)
        else:
            self.bw = self.bw_method

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
        if not hasattr(self, "data"):
            raise ValueError("Must call fit before evaluating.")

        # -------------- Set up the bandwidth depending on inputs -------------
        if bw_to_scalar:
            bw = np.max(self.bw)
        else:
            bw = self.bw

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
            raise ValueError("Grid must contain at least one data point.")
        self.grid_points = grid_points

        # Test quickly that the method has done what is was supposed to do
        if bw_to_scalar:
            assert isinstance(bw, numbers.Number)
            assert bw > 0
        assert len(self.grid_points.shape) == 2

    def score(self, test_data, test_weights=None):
        """
        Computes the score of test data on the KDE model. The score is
        calculated as the mean log-probability of the test samples
        on the model. The method takes into account test weights, and
        works with variable bandwidths.

        Parameters
        ----------
        test_data : array-like or Sequence
            May be array-like of shape (obs,), shape (obs, dims) or a
            Python Sequence, e.g. a list or tuple.
        test_weights : array-like, Sequence or None
            May be array-like of shape (obs,), shape (obs, dims), a
            Python Sequence, e.g. a list or tuple, or None.
        """

        # -------------- Set up the data depending on input -------------------
        # In the end, the data should be an ndarray of shape (obs, dims)
        test_data = self._process_sequence(test_data)

        obs, dims = test_data.shape

        if not obs > 0:
            raise ValueError("Test data must contain at least one data point.")
        assert dims > 0

        # -------------- Set up the weights depending on input ----------------
        if test_weights is not None:
            test_weights = self._process_sequence(test_weights).ravel()
            if not obs == len(test_weights):
                raise ValueError("Number of test data obs must match test weights")

            return np.mean(test_weights * np.log(self.evaluate(test_data)))

        return np.mean(np.log(self.evaluate(test_data)))

    def cross_val(self, data, weights=None, cv=10, seed=None, grid=None):
        """
        Computes the cross validated score over a grid of bandwidths, and returns
        the one that maximizes it. It is a robust method against multimodal
        distributions, and can be performed on variable bandwidths (e.g.: by
        setting "seed" parameter as the output of k nearest neighbors algorithm).

        Habbema, J. D. F., Hermans, J., and Van den Broek, K. (1974) A stepwise
        discrimination analysis program using density estimation.

        Leave-one-out MLCV method in R: https://rdrr.io/cran/kedd/man/h.mlcv.html

        Parameters
        ----------
        data: array-like
            The data points. Data must have shape (obs, dims).
        weights: array-like,
            One weight per data point. Numbers of observations must match
            the data points.
        cv: int
            The number of cross validation folds. If cv equals obs, it is the
            leave-one-out cross validation.
        seed : float or array-like
            The seed bandwidth. By default is a simplified version of the silverman
            rule.
        grid : array-like
            The grid of factors. The bandwidth grid is constructed as:
            bw_grid[i] = bw * grid[i]
            By default is np.logspace(-1,1,20)
        """
        return cross_val(self, data, weights=weights, cv=cv, seed=seed, grid=grid)

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
                raise ValueError("Must be of shape (obs, dims)")
        else:
            raise TypeError("Must be of shape (obs, dims)")
        return np.asarray_chkfinite(out, dtype=float)

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
    pytest.main(args=[".", "--doctest-modules", "-v", "-x"])
