#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for bandwidth selection.
"""
import numbers
import numpy as np
import scipy
import warnings
from collections.abc import Sequence
from KDEpy.binning import linear_binning
from KDEpy.utils import autogrid
from scipy import fftpack
from scipy.optimize import brentq
from scipy.spatial import KDTree

# Choose the largest available float on the system
try:
    FLOAT = scipy.float128
except AttributeError:
    FLOAT = np.float64


def _fixed_point(t, N, I_sq, a2):
    r"""
    Compute the fixed point as described in the paper by Botev et al.

    .. math:

        t = \xi \gamma^{5}(t)

    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    I_sq : array-like
        The numbers [1, 2, 9, 16, ...]
    a2 : array-like
        The DCT of the original data, divided by 2 and squared.

    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01, 50, np.arange(1, 51), np.arange(1, 51))
    >>> assert np.allclose(ans, 0.0099076220293967618515)
    >>> # another
    >>> ans = _fixed_point(0.07, 25, np.arange(1, 11), np.arange(1, 11))
    >>> assert np.allclose(ans, 0.068342291525717486795)

    References
    ----------
     - Implementation by Daniel B. Smith, PhD, found at
       https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    """

    # This is important, as the powers might overflow if not done
    I_sq = np.asfarray(I_sq, dtype=FLOAT)
    a2 = np.asfarray(a2, dtype=FLOAT)

    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = 7

    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = (0.5) * np.pi ** (2 * ell) * np.sum(np.power(I_sq, ell) * a2 * np.exp(-I_sq * np.pi ** 2 * t))

    # Norm of a function, should never be negative
    if f <= 0:
        return -1
    for s in reversed(range(2, ell)):
        # This could also be formulated using the double factorial n!!,
        # but this is faster so and requires an import less

        # Step one: estimate t_s from |f^(s+1)|^2
        odd_numbers_prod = np.product(np.arange(1, 2 * s + 1, 2, dtype=FLOAT))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))

        # Step two: estimate |f^s| from t_s
        f = (0.5) * np.pi ** (2 * s) * np.sum(np.power(I_sq, s) * a2 * np.exp(-I_sq * np.pi ** 2 * time))

    # This is the minimizer of the AMISE
    t_opt = np.power(2 * N * np.sqrt(np.pi) * f, -2.0 / 5)

    # Return the difference between the original t and the optimal value
    return t - t_opt


def _root(function, N, args):
    """
    Root finding algorithm. Based on MATLAB implementation by Botev et al.

    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> np.allclose(ans, 9.237610787616029e-05)
    True
    """
    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000
    # While a solution is not found, increase the tolerance and try again
    found = 0
    while found == 0:
        try:
            # Other viable solvers include: [brentq, brenth, ridder, bisect]
            x, res = brentq(function, 0, tol, args=args, full_output=True, disp=False)
            found = 1 if res.converged else 0
        except ValueError:
            x = 0
            tol *= 2.0
            found = 0
        if x <= 0:
            found = 0

        # If the tolerance grows too large, minimize the function
        if tol >= 1:
            raise ValueError("Root finding did not converge. Need more data.")

    if not x > 0:
        raise ValueError("Root finding failed to find positive solution.")
    return x


def improved_sheather_jones(data, weights=None):
    """
    The Improved Sheater Jones (ISJ) algorithm from the paper by Botev et al.
    This algorithm computes the optimal bandwidth for a gaussian kernel,
    and works very well for bimodal data (unlike other rules). The
    disadvantage of this algorithm is longer computation time, and the fact
    that this implementation does not always converge if very few data
    points are supplied.

    Understanding this algorithm is difficult, see:
    https://books.google.no/books?id=Trj9HQ7G8TUC&pg=PA328&lpg=PA328&dq=
    sheather+jones+why+use+dct&source=bl&ots=1ETdKd_6EF&sig=jZk4R515GB1xsn-
    VZVnjr-JfjSI&hl=en&sa=X&ved=2ahUKEwi1_czNncTcAhVGhqYKHaPiBtcQ6AEwA3oEC
    AcQAQ#v=onepage&q=sheather%20jones%20why%20use%20dct&f=false

    Parameters
    ----------
    data: array-like
        The data points. Data must have shape (obs, 1).
    weights: array-like, optional
        One weight per data point. Must have shape (obs,). If None is
        passed, uniform weights are used.
    """
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("ISJ is only available for 1D data.")

    n = 2 ** 10

    # weights <= 0 still affect calculations unless we remove them
    if weights is not None:
        data = data[weights > 0]
        weights = weights[weights > 0]

    # Setting `percentile` higher decreases the chance of overflow
    xmesh = autogrid(data, boundary_abs=6, num_points=n, boundary_rel=0.5)
    data = data.ravel()
    xmesh = xmesh.ravel()

    # Create an equidistant grid
    R = np.max(data) - np.min(data)
    # dx = R / (n - 1)
    data = data.ravel()
    N = len(np.unique(data))

    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    initial_data = linear_binning(data.reshape(-1, 1), xmesh, weights)
    assert np.allclose(initial_data.sum(), 1)

    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data)

    # Compute the bandwidth
    # The definition of a2 used here and in `_fixed_point` correspond to
    # the one cited in this issue:
    # https://github.com/tommyod/KDEpy/issues/95
    I_sq = np.power(np.arange(1, n, dtype=FLOAT), 2)
    a2 = a[1:] ** 2

    # Solve for the optimal (in the AMISE sense) t
    t_star = _root(_fixed_point, N, args=(N, I_sq, a2))

    # The remainder of the algorithm computes the actual density
    # estimate, but this function is only used to compute the
    # bandwidth, since the bandwidth may be used for other kernels
    # apart from the Gaussian kernel

    # Smooth the initial data using the computed optimal t
    # Multiplication in frequency domain is convolution
    # integers = np.arange(n, dtype=float)
    # a_t = a * np.exp(-integers**2 * np.pi ** 2 * t_star / 2)

    # Diving by 2 done because of the implementation of fftpack.idct
    # density = fftpack.idct(a_t) / (2 * R)

    # Due to overflow, some values might be smaller than zero, correct it
    # density[density < 0] = 0.
    bandwidth = np.sqrt(t_star) * R
    return bandwidth


def scotts_rule(data, weights=None):
    """
    Scotts rule.

    Scott (1992, page 152)
    Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and
    Visualization. New York: Wiley.

    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> ans = scotts_rule(data)
    >>> assert np.allclose(ans, 1.76474568962182)
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")

    if weights is not None:
        warnings.warn("Scott's rule currently ignores all weights")

    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("Scotts rule is only available for 1D data.")
    sigma = np.std(data, ddof=1)
    # scipy.norm.ppf(.75) - scipy.norm.ppf(.25) -> 1.3489795003921634
    IQR = (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634

    sigma = min(sigma, IQR)
    return sigma * np.power(obs, -1.0 / (dims + 4))


def silvermans_rule(data, weights=None):
    """
    Returns optimal smoothing (standard deviation) if the data is close to
    normal.

    TODO: Extend to multidimensional:
        https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.
        stats.gaussian_kde.html#r216

    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> ans = silvermans_rule(data)
    >>> assert np.allclose(ans, 1.8692607078355594)
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("Silverman's rule is only available for 1D data.")

    if weights is not None:
        warnings.warn("Silverman's rule currently ignores all weights")

    if obs == 1:
        return 1
    if obs < 1:
        raise ValueError("Data must be of length > 0.")

    sigma = np.std(data, ddof=1)
    # scipy.stats.norm.ppf(.75) - scipy.stats.norm.ppf(.25) -> 1.3489795003921634
    IQR = (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634

    sigma = min(sigma, IQR)

    # The logic below is not related to silverman's rule, but if the data is constant
    # it's nice to return a value instead of getting an error. A warning will be raised.
    if sigma > 0:
        return sigma * (obs * 3 / 4.0) ** (-1 / 5)
    else:
        # stats.norm.ppf(.99) - stats.norm.ppf(.01) = 4.6526957480816815
        IQR = (np.percentile(data, q=99) - np.percentile(data, q=1)) / 4.6526957480816815
        if IQR > 0:
            bw = IQR * (obs * 3 / 4.0) ** (-1 / 5)
            warnings.warn(
                "Silverman's rule failed. Too many idential values. \
Setting bw = {}".format(
                    bw
                )
            )
            return bw

        # Here, all values are basically constant
        warnings.warn("Silverman's rule failed. Too many idential values. Setting bw = 1.0")
        return 1.0


def k_nearest_neighbors(data, weights=None, k=10, batch_size=10000):
    """
    Computes variable bandwidth based on k nearest neighbors algorithm on data.
    For each data point, sets its bandwidth as the euclidean distance to the
    kth nearest neighbor within its batch. The computation is performed on
    batches to allow scalability to large datasets. The scipy KDTree class is
    used for the nearest neighbors queries.

    https://en.wikipedia.org/wiki/Variable_kernel_density_estimation
    https://en.wikipedia.org/wiki/K-nearest_neighbour_algorithm
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html

    Parameters
    ----------
    data: array-like
        The data points. Data must have shape (obs, dims).
    weights: array-like, 
        Ignored, only for compatibility
    k: int
        Number of neighbors per batch (without counting self)
    batch_size: int
        Aproximate size of each batch. Will be slightly modified to cover
        dataset as uniformly as possible.
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape

    if weights is not None:
        warnings.warn("K nearest neighbors ignores all weights")

    if obs < 2:
        raise ValueError("Data must be of length >= 2.")

    if k < 1 or k >= batch_size:
        raise ValueError("k must be between 1 and batch_size-1.")
    k = int(k)

    if batch_size < 2:
        raise ValueError("Batch size must be >= 2.")

    batches = int(max(1, np.round(obs / batch_size)))
    batch_size = int(obs / batches)
    print("Using k = {} neighbors per batch (batch size = {})".format(k, batch_size))
    print("Equivalent to aprox. {} total neighbors".format(k*batches))

    # This could be easily run in parallel, improving performance, but it would
    # require depending on an external library (e.g.: joblib)
    bw_knn = np.array([])
    for batch,batch_data in enumerate(np.array_split(data, batches)):
        print("K Nearest Neighbors: batch = {} of {}".format(batch+1, batches))
        kdtree = KDTree(batch_data)
        # Use k+1 to take into account dist=0 between each point and self
        dists,idxs = kdtree.query(batch_data, k=k+1)
        bw_knn = np.concatenate((bw_knn, dists[:,-1]))
    return bw_knn


def _cv_score(model, bw, data, weights=None, cv=10):
    """
    Computes cross validated score of KDE model, which gives an indicator of
    the quality of the estimation. Test and train data are taken following a
    K-folds cross validation scheme.

    Parameters
    ----------
    model: NaiveKDE() or TreeKDE()
        The model to be used to evaluate scores. Technically it could be any
        object that implements methods __init__, fit and evaluate (on arbitrary
        grid) with the same syntax than KDEpy.
    bw : float or array-like
        Bandwidth. If a float is passed, it is the standard deviation of the
        kernel. If an array-like it passed, it is the bandwidth of each point.
    data: array-like
        The data points. Data must have shape (obs, dims).
    weights: array-like, 
        One weight per data point. Numbers of observations must match
        the data points.
    cv: int
        The number of cross validation folds. If cv equals obs, it is the
        leave-one-out cross validation.
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape

    if obs < 2:
        raise ValueError("Data must be of length >= 2.")

    if weights is not None:
        if not weights.shape == (obs,):
            raise ValueError("Number of weights must match data points")

    if cv > len(data):
        cv = len(data)
    if cv < 2:
        raise ValueError("Number of folds must be >= 2.")

    if isinstance(bw, numbers.Number) and bw > 0:
        variable_bw = False
    elif isinstance(bw, (np.ndarray, Sequence)):
        if not bw.shape == (obs,):
            raise ValueError("If bandwidth is variable, it must match data points")
        folds_bw = np.array_split(bw, cv)
        variable_bw = True
    else:
        raise ValueError("Bandwidth must be > 0, or array-like.")

    # Folds for cross validation
    folds_data = np.array_split(data, cv)
    if weights is not None:
        folds_weights = np.array_split(weights, cv)
    else:
        folds_weights = cv * [None]
    folds_score = []

    # Compute cross validation for each fold
    for fold,[test_data,test_weights] in enumerate(zip(folds_data,folds_weights)):
        if variable_bw:
            train_bw = np.concatenate(folds_bw[:fold]+folds_bw[fold+1:], axis=0)
        else:
            train_bw = bw
        kde = model.__class__(kernel=model.kernel, bw=train_bw, norm=model.norm)
        train_data = np.concatenate(folds_data[:fold]+folds_data[fold+1:], axis=0)
        if weights is not None:
            train_weights = np.concatenate(folds_weights[:fold]+folds_weights[fold+1:], axis=0)
        else:
            train_weights = None
        kde.fit(train_data, weights=train_weights)
        folds_score.append(kde.score(test_data,test_weights))

    return np.mean(folds_score)


def grid_search_cv(model, bw_grid, data, weights=None, cv=10):
    """
    Computes the cross validated score over a grid of bandwidths, and returns
    the list of cv scores.

    Parameters
    ----------
    model: NaiveKDE() or TreeKDE()
        The model to be used to evaluate scores. Technically it could be any
        object that implements methods __init__, fit and evaluate (on arbitrary
        grid) with the same syntax than KDEpy.
    bw_grid : array-like
        The grid of bandwidths. Each element can be a float or an array of
        shape (obs,).
    data: array-like
        The data points. Data must have shape (obs, dims).
    weights: array-like, 
        One weight per data point. Numbers of observations must match
        the data points.
    cv: int
        The number of cross validation folds. If cv equals obs, it is the
        leave-one-out cross validation.
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape

    if obs < 2:
        raise ValueError("Data must be of length >= 2.")

    assert isinstance(bw_grid, (Sequence, np.ndarray))

    # This could be easily run in parallel, improving performance, but it would
    # require depending on an external library (e.g.: joblib)
    cv_scores = []
    for idx,bw in enumerate(bw_grid):
        print("Cross Validation: evaluating bandwidth {} of {}".format(idx+1, len(bw_grid)))
        cv_scores.append(_cv_score(model, bw, data, weights=weights, cv=cv))
    
    return cv_scores


def cross_val(model, data, weights=None, cv=10, seed=None, grid=None):
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
    model: NaiveKDE() or TreeKDE()
        The model to be used to evaluate scores. Technically it could be any
        object that implements methods __init__, fit and evaluate (on arbitrary
        grid) with the same syntax than KDEpy.
    bw : float or array-like
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
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape

    if obs < 2:
        raise ValueError("Data must be of length >= 2.")

    if seed is None:
        # This should be replaced by a call to silverman_rule when it is
        # implemented for multidimensional data
        sigma = np.std(data, axis=0).mean()
        seed = sigma * (obs * (2.0+dims/4)) ** (-1/(4+dims))

    if grid is None:
        grid = np.logspace(-1, 1, 20)

    bw_grid = np.reshape(grid, (-1,*np.ones(np.ndim(seed),dtype=int))) * seed

    cv_scores = grid_search_cv(model, bw_grid, data, weights=weights, cv=cv)
    idx_best = np.argmax(cv_scores)

    # Warn if maximum was in beginning or end of grid
    if idx_best in (0, len(bw_grid)-1):
        # Could calculate new grid automatically and call recursively
        msg = "Could not find maximum in the selected range of bandwidths.\n"
        msg += "Move grid and try again."
        warnings.warn(msg)

    bw_cv = bw_grid[idx_best]
    return bw_cv


_bw_methods = {
    "silverman": silvermans_rule,
    "scott": scotts_rule,
    "ISJ": improved_sheather_jones,
    "knn": k_nearest_neighbors,
    # CV can not be added here since it requires a model
}
