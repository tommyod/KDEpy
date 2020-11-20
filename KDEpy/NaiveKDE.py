#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the NaiveKDE.
"""
import numbers
import itertools
import numpy as np
from KDEpy.BaseKDE import BaseKDE


class NaiveKDE(BaseKDE):
    """
    This class implements a naive computation of a kernel density estimate. The
    advantages are that choices of bandwidth, norms, weights and grids are
    straightforward -- the user can do almost anything. The disadvantage is
    that computations are slow on more than a couple of thousand data points.

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
    >>> x, y = NaiveKDE(bw='ISJ').fit(data).evaluate()
    >>> # (2) Explicit choice of kernel and bw (standard deviation of kernel)
    >>> x, y = NaiveKDE(kernel='triweight', bw=0.5).fit(data).evaluate()
    >>> weights = data + 10
    >>> # (3) Using a grid and weights for the data
    >>> y = NaiveKDE(kernel='epa', bw=0.5).fit(data, weights).evaluate(x)

    References
    ----------
    - Silverman, B. W. Density Estimation for Statistics and Data Analysis.
      Boca Raton: Chapman and Hall, 1986.
    - Wand, M. P., and M. C. Jones. Kernel Smoothing.
      London ; New York: Chapman and Hall/CRC, 1995.
    - Scipy implementation, at ``scipy.stats.gaussian_kde``.
    """

    def __init__(self, kernel="gaussian", bw=1, norm=2):
        super().__init__(kernel, bw)
        self.norm = norm

    def fit(self, data, weights=None):
        """
        Fit the KDE to the data. This validates the data and stores it.
        Computations are performed when the KDE is evaluated on a grid.

        Parameters
        ----------
        data: array-like
            The data points. High dimensional data must have shape (obs, dims).
        weights: array-like
            One weight per data point. Must have shape (obs,). If None is
            passed, uniform weights are used.

        Returns
        -------
        self
            Returns the instance.

        Examples
        --------
        >>> data = [1, 3, 4, 7]
        >>> weights = [3, 4, 2, 1]
        >>> kde = NaiveKDE().fit(data, weights=None)
        >>> kde = NaiveKDE().fit(data, weights=weights)
        >>> x, y = kde()
        """
        # Sets self.data
        super().fit(data, weights)
        return self

    def evaluate(self, grid_points=None):
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

        Returns
        -------
        y: array-like
            If a grid is supplied, `y` is returned. If no grid is supplied,
            a tuple (`x`, `y`) is returned.

        Examples
        --------
        >>> kde = NaiveKDE().fit([1, 3, 4, 7])
        >>> # Two ways to evaluate, either with a grid or without
        >>> x, y = kde.evaluate()
        >>> x, y = kde.evaluate(256)
        >>> y = kde.evaluate(x)
        """
        # This method sets self.grid points and verifies it
        # NaiveKDE does not convert the bw to a scalar, since a vector is
        # allowed too.
        super().evaluate(grid_points, bw_to_scalar=False)

        # Create zeros on the grid points
        evaluated = np.zeros(self.grid_points.shape[0])

        # For every data point, compute the kernel and add to the grid
        bw = self.bw
        if isinstance(bw, numbers.Number):
            bw = np.asfarray(np.ones(self.data.shape[0]) * bw)

        # TODO: Implementation w.r.t grid points for faster evaluation
        # See the SciPy evaluation for how this can be done
        weights = itertools.repeat(1 / self.data.shape[0]) if self.weights is None else self.weights

        for weight, data_point, bw in zip(weights, self.data, bw):
            x = self.grid_points - data_point
            evaluated += weight * self.kernel(x, bw=bw, norm=self.norm)

        return self._evalate_return_logic(evaluated, self.grid_points)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v"])
