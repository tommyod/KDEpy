#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the API. The API should be considered fixed in future releases and should
be equal for every implementation. Therefore it's important to have unit tests
for the API.
"""
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.TreeKDE import TreeKDE
import itertools
import pytest
import matplotlib

matplotlib.use("Agg")  # For testing on servers

kernels = list(NaiveKDE._available_kernels.keys())
kdes = [NaiveKDE, TreeKDE, FFTKDE]
kde_pairs = list(itertools.combinations(kdes, 2))


@pytest.mark.parametrize(
    "kde1, kde2, bw, kernel",
    [(k[0], k[1], bw, ker) for (k, bw, ker) in itertools.product(kde_pairs, [0.1, "silverman", 1], kernels)],
)
def test_api_models_kernels_bandwidths(kde1, kde2, bw, kernel):
    """
    Test the API. More specifically the chained version and the non-chained
    version of the API. It's tested over every implementation, several
    bandwidths and kernels.
    """
    data = np.array([-1, 0, 0.1, 3, 10])
    weights = [1, 2, 1, 0.8, 2]

    # Chained expression
    x1, y1 = kde1(kernel=kernel, bw=bw).fit(data, weights).evaluate()

    # Step by step, with previous grid
    model = kde2(kernel=kernel, bw=bw)
    model.fit(data, weights)
    y2 = model.evaluate(x1)

    # Mean error
    err = np.sqrt(np.mean((y1 - y2) ** 2))
    if kernel == "box":
        assert err < 0.025
    else:
        assert err < 0.002


type_functions = [tuple, np.array, np.asfarray, lambda x: np.asfarray(x).reshape(-1, 1)]


@pytest.mark.parametrize(
    "kde, bw, kernel, type_func",
    itertools.product(kdes, ["silverman", "scott", "ISJ", 0.5], ["epa", "gaussian"], type_functions),
)
def test_api_types(kde, bw, kernel, type_func):
    """
    Test the API. Data and weights may be passed as tuples, arrays, lists, etc.
    """
    # Test various input types
    data = np.random.randn(64)
    weights = np.random.randn(64) + 10
    model = kde(kernel=kernel, bw=bw)
    x, y = model.fit(data, weights).evaluate()

    data = type_func(data)
    weights = type_func(weights)
    y1 = model.fit(data, weights).evaluate(x)
    assert np.allclose(y, y1)


@pytest.mark.parametrize(
    "kde1, kde2, bw, kernel",
    [(k[0], k[1], bw, ker) for (k, bw, ker) in itertools.product(kde_pairs, [0.5, 1], kernels)],
)
def test_api_models_kernels_bandwidths_2D(kde1, kde2, bw, kernel):
    """
    Test the API on 2D data.
    """

    data = np.array([[0, 0], [0, 1], [0, 0.5], [-1, 1]])
    weights = [1, 2, 1, 0.8]

    points = 2 ** 5

    # Chained expression
    x1, y1 = kde1(kernel=kernel, bw=bw).fit(data, weights).evaluate(points)

    # Step by step, with previous grid
    model = kde2(kernel=kernel, bw=bw)
    model.fit(data, np.array(weights))
    x2, y2 = model.evaluate(points)

    # Mean error
    err = np.sqrt(np.mean((y1 - y2) ** 2))
    if kernel in ("box", "logistic", "sigmoid"):
        assert True
    else:
        assert err < 0.0025


@pytest.mark.parametrize("estimator", kdes)
def test_api_2D_data(estimator):
    """
    Test a plotting example on 2D data. Tested over every KDE implementation.
    This test specifices how plotting may be done with 2D KDEs.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Create 2D data of shape (obs, dims)
    np.random.seed(123)
    n = 16
    data = np.concatenate((np.random.randn(n).reshape(-1, 1), np.random.randn(n).reshape(-1, 1)), axis=1)

    grid_points = 2 ** 5  # Grid points in each dimension
    N = 16  # Number of contours

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))

    for ax, norm in zip(axes, [1, 2, np.inf]):

        ax.set_title("Norm $p={}$".format(norm))

        # Compute
        kde = estimator(kernel="gaussian", norm=norm)
        grid, points = kde.fit(data).evaluate(grid_points)

        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(grid_points, grid_points).T

        # Plot the kernel density estimate
        ax.contour(x, y, z, N, linewidths=0.8, colors="k")
        ax.contourf(x, y, z, N, cmap="RdBu_r")
        ax.plot(data[:, 0], data[:, 1], "ok", ms=3)

    plt.tight_layout()
    plt.close()


@pytest.mark.parametrize("estimator", kdes)
def test_api_2D_data_which_is_1D(estimator):
    """
    Test that 2D data along a line is proportionally the same as 1D data.
    It's not identical since the integrals must evaluate to unity.
    """

    np.random.seed(123)
    random_data = np.random.randn(50).reshape(-1, 1)
    zeros = np.zeros_like(random_data)
    data_2D = np.concatenate((random_data, zeros), axis=1)

    x2, y2 = NaiveKDE().fit(data_2D).evaluate((1024, 3))
    y2 = y2.reshape((1024, 3))
    x, y = NaiveKDE().fit(random_data).evaluate(1024)

    # Proportions
    prop = y2[:, 3 // 2].ravel() / y

    # At zero, epsilon is added and eps / eps = 1, remove these values
    prop = prop[~np.isclose(prop, 1)]

    # Every other value should be equal - i.e they should be proportional
    # To see why they are equal, consider points (0, 0), (1, 0) and (2, 0).
    # Depending on the norm the normalization will make the heigh smaller
    assert np.all(np.isclose(prop, prop[0]))

    # Again the other way around too
    data_2D = np.concatenate((zeros, random_data), axis=1)
    x2, y2 = NaiveKDE().fit(data_2D).evaluate((3, 1024))
    y2 = y2.reshape((3, 1024))
    x, y = NaiveKDE().fit(random_data).evaluate(1024)
    prop = y2[3 // 2, :].ravel() / y
    prop = prop[~np.isclose(prop, 1)]
    assert np.all(np.isclose(prop, prop[0]))


@pytest.mark.parametrize("estimator", kdes)
def test_fitting_twice(estimator):
    """Fitting several times should re-fit the BW.
    Issue: https://github.com/tommyod/KDEpy/issues/78
    """
    x_grid = np.linspace(-100, 100, 2 ** 6)

    # Create two data sets
    data = np.arange(-5, 6)
    data_scaled = data * 10

    kde = estimator(bw="silverman")

    # The BW from the first fit should be used in the second fit
    kde.fit(data).evaluate(x_grid)
    y = kde.fit(data_scaled).evaluate(x_grid)

    y2 = estimator(bw="silverman").fit(data_scaled).evaluate(x_grid)

    assert np.allclose(y, y2)


if __name__ == "__main__":
    if True:
        # --durations=10  <- May be used to show potentially slow tests
        pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-k fitting"])
