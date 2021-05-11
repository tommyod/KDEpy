#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the TreeKDE.
"""
from scipy.spatial import cKDTree
import numbers
import numpy as np
from KDEpy.BaseKDE import BaseKDE


class TreeKDE(BaseKDE):
    """
    This class implements a tree-based computation of a kernel density
    estimate. It works by segmenting the space recursively into smaller parts.

    This makes computing a kernel density estimate at a location easier, since
    we are able to query the tree structure for nearby points instead of having
    to evaluate the kernel function on all data points. For kernels without
    finite support, their support is approximated. The ``scipy`` k-d tree is
    used as the underlying algorithm.

    Parameters
    ----------
    kernel : str
        The kernel function. See cls._available_kernels.keys() for choices.
    bw : float, str or array-like
        Bandwidth or bandwidth selection method. If a float is passed, it
        is the standard deviation of the kernel. If a string it passed, it
        is the bandwidth selection method, see cls._bw_methods.keys() for
        choices. If an array-like it passed, it is the bandwidth of each
        point.
    norm : float
        The p-norm used to compute the distances in higher dimensions.

    Examples
    --------
    >>> data = np.random.randn(2**10)
    >>> # (1) Automatic bw selection using Improved Sheather Jones
    >>> x, y = TreeKDE(bw='ISJ').fit(data).evaluate()
    >>> # (2) Explicit choice of kernel and bw (standard deviation of kernel)
    >>> x, y = TreeKDE(kernel='triweight', bw=0.5).fit(data).evaluate()
    >>> weights = data + 10
    >>> # (3) Using a grid and weights for the data
    >>> y = TreeKDE(kernel='epa', bw=0.5).fit(data, weights).evaluate(x)

    References
    ----------
    - Friedman, Jerome H., Jon Louis Bentley, and Raphael Ari Finkel.
      An Algorithm for Finding Best Matches in Logarithmic Expected Time.
      ACM Trans. Math. Softw. 3, no. 3 (September 1977): 209–226.
      https://doi.org/10.1145/355744.355745.
    - Maneewongvatana, Songrit, and David M. Mount.
      It’s Okay to Be Skinny, If Your Friends Are Fat.
      In Center for Geometric Computing 4th Annual Workshop on Computational
      Geometry, 2:1–8, 1999.
    - Silverman, B. W. Density Estimation for Statistics and Data Analysis.
      Boca Raton: Chapman and Hall, 1986. Page 99 for reference to kd-tree.
    - Scipy implementation, at ``scipy.spatial.KDTree``.
    """

    def __init__(self, kernel="gaussian", bw=1, norm=2.0):
        super().__init__(kernel, bw)
        self.norm = norm

    def fit(self, data, weights=None):
        """
        Fit the KDE to the data. This validates the data and stores it.
        Computations are performed upon evaluation on a grid.

        Parameters
        ----------
        data: array-like
            The data points.
        weights: array-like
            One weight per data point. Numbers of observations must match
            the data points.

        Returns
        -------
        self
            Returns the instance.

        Examples
        --------
        >>> data = [1, 3, 4, 7]
        >>> weights = [3, 4, 2, 1]
        >>> kde = TreeKDE().fit(data, weights=None)
        >>> kde = TreeKDE().fit(data, weights=weights)
        >>> x, y = kde()
        """
        # Sets self.data
        super().fit(data, weights)
        return self

    def evaluate(self, grid_points=None, eps=10e-4):
        """
        Evaluate on grid points.

        Parameters
        ----------
        grid_points: array-like, int, tuple or None
            A grid (mesh) to evaluate on. High dimensional grids must have
            shape (obs, dims). If an integer is passed, it's the number of grid
            points on an equidistant grid. If a tuple is passed, it's the
            number of grid points in each dimension. If None, a grid will be
            automatically created.
        eps: float
            The maximal total error in absolute terms when estimating the
            effective support of a kernel which has infinite support. Setting
            this too high will produced a jagged estimate.

        Returns
        -------
        y: array-like
            If a grid is supplied, `y` is returned. If no grid is supplied,
            a tuple (`x`, `y`) is returned.

        Examples
        --------
        >>> kde = TreeKDE().fit([1, 3, 4, 7])
        >>> # Two ways to evaluate, either with a grid or without
        >>> x, y = kde.evaluate()
        >>> x, y = kde.evaluate(256)
        >>> y = kde.evaluate(x)
        """

        # This method sets self.grid points and verifies it
        super().evaluate(grid_points)

        evaluated = np.zeros(self.grid_points.shape[0])

        # For every data point, compute the kernel and add to the grid
        obs, dims = self.data.shape
        bw = self.bw
        if isinstance(bw, numbers.Number):
            bw = np.asfarray(np.ones(obs) * bw)
        else:
            bw = np.asarray_chkfinite(bw, dtype=float)

        # Initialize the tree structure for fast lookups of neighbors
        tree = cKDTree(self.data)

        # Compute the kernel radius
        maximal_bw = np.max(bw)
        if not eps > 0:
            raise ValueError("eps must be > 0.")
        kernel_radius = self.kernel.practical_support(maximal_bw, eps)

        # Since we iterate through grid points, we need the maximum bw to
        # ensure that we get data points that are close enough
        for i, grid_point in enumerate(self.grid_points):

            # Query for data points that are close to this grid point
            # TODO: Is this epsilon value sensible?
            # Scipy 1.3.0 introduced error: ValueError: ndarray is not C-contiguous
            grid_point = np.ascontiguousarray(grid_point)
            indices = tree.query_ball_point(x=grid_point, r=kernel_radius, p=self.norm, eps=eps * obs ** 0.5)

            # Use broadcasting to find x-values (distances)
            x = grid_point - self.data[indices]
            kernel_estimates = self.kernel(x, bw=bw[indices], norm=self.norm)

            if self.weights is not None:
                weights_subset = self.weights[indices]
                assert kernel_estimates.shape == weights_subset.shape
            assert kernel_estimates.shape == bw[indices].shape

            # Unpack the (n, 1) arrays to (n,) and compute the doc product
            if self.weights is not None:
                evaluated[i] += np.dot(kernel_estimates, weights_subset)
            else:
                evaluated[i] += np.sum(kernel_estimates) / obs

        return self._evalate_return_logic(evaluated, self.grid_points)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v"])
