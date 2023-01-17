#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the FFTKDE.
"""
import itertools

import numpy as np
import pytest

from KDEpy.FFTKDE import FFTKDE
from KDEpy.NaiveKDE import NaiveKDE

args = list(itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2]], [1, 2, 3]))


@pytest.mark.parametrize("data, split_index", args)
def test_additivity(data, split_index):
    """
    Test the additive propery of the KDE.

    TODO: Parameterize this test w.r.t implementation.
    """
    x = np.linspace(-10, 12)

    # Fit to add data
    y = FFTKDE("epa").fit(data).evaluate(x)

    # Fit to splits, and compensate for smaller data using weights
    weight_1 = split_index / len(data)
    y_1 = FFTKDE("epa").fit(data[:split_index]).evaluate(x) * weight_1

    weight_2 = (len(data) - split_index) / len(data)
    y_2 = FFTKDE("epa").fit(data[split_index:]).evaluate(x) * weight_2

    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)


@pytest.mark.parametrize("data, split_index", args)
def test_additivity_with_weights(data, split_index):
    """
    Test the additive propery of the KDE.

    TODO: Parameterize this test w.r.t implementation.
    """

    x = np.linspace(-10, 15)
    weights = np.arange(len(data)) + 1
    weights = weights / np.sum(weights)

    # Fit to add data
    y = FFTKDE("epa").fit(data, weights).evaluate(x)

    # Split up the data and the weights
    data = list(data)
    weights = list(weights)
    data_first_split = data[:split_index]
    data_second_split = data[split_index:]
    weights_first_split = weights[:split_index]
    weights_second_split = weights[split_index:]

    # Fit to splits, and compensate for smaller data using weights
    y_1 = FFTKDE("epa").fit(data_first_split, weights_first_split).evaluate(x) * sum(weights_first_split)

    y_2 = FFTKDE("epa").fit(data_second_split, weights_second_split).evaluate(x) * sum(weights_second_split)

    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)


args = itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2], [0.1]], [1, 2, 3])


@pytest.mark.parametrize("data, bw", args)
def test_against_naive_KDE(data, bw):
    """
    The the FFTKDE against a naive KDE without weights.
    """

    # Higher accuracy when num gets larger
    x = np.linspace(min(data) - bw, max(data) + bw, num=2**10)

    y1 = NaiveKDE("epa", bw=bw).fit(data, weights=None).evaluate(x)
    y2 = FFTKDE("epa", bw=bw).fit(data, weights=None).evaluate(x)

    assert np.allclose(y1, y2, atol=10e-5)


args = itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2], [0.1]], [1, 2, 3])


@pytest.mark.parametrize("data, bw", args)
def test_against_naive_KDE_w_weights(data, bw):
    """
    The the FFTKDE against a naive KDE with weights.
    """

    # Higher accuracy when num gets larger
    x = np.linspace(min(data) - bw, max(data) + bw, num=2**10)
    weights = np.arange(len(data)) + 1

    y1 = NaiveKDE("epa", bw=bw).fit(data, weights=weights).evaluate(x)
    y2 = FFTKDE("epa", bw=bw).fit(data, weights=weights).evaluate(x)

    assert np.allclose(y1, y2, atol=10e-4)


def FFTKDE_test_grid_inside_data_1D():
    """
    When using a custom grid, an error should be raised if the data is not
    contained in the grid. The linear binning routine will crash if this
    is not the case. See Issue:
    https://github.com/tommyod/KDEpy/issues/7
    """
    data = np.array([0, 1, 2, 3, 4, 5])
    grid = np.linspace(-1, 6, num=2**6)
    FFTKDE().fit(data).evaluate(grid)  # This should cause no problem

    with pytest.raises(ValueError):
        bad_grid = np.linspace(2, 6, num=2**6)
        FFTKDE().fit(data).evaluate(bad_grid)

    with pytest.raises(ValueError):
        bad_grid = np.linspace(-2, 4, num=2**6)
        FFTKDE().fit(data).evaluate(bad_grid)

    with pytest.raises(ValueError):
        bad_grid = np.linspace(0, 5, num=2**6)
        FFTKDE().fit(data).evaluate(bad_grid)


def FFTKDE_test_grid_inside_data_2D():
    """
    When using a custom grid, an error should be raised if the data is not
    contained in the grid. The linear binning routine will crash if this
    is not the case. See Issue:
    https://github.com/tommyod/KDEpy/issues/7
    """
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    grid, y = FFTKDE().fit(data).evaluate()  # To get a grid

    with pytest.raises(ValueError):
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 6]])
        FFTKDE().fit(data).evaluate(grid)

    with pytest.raises(ValueError):
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, -4]])
        FFTKDE().fit(data).evaluate(grid)

    with pytest.raises(ValueError):
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 100]])
        FFTKDE().fit(data).evaluate(grid)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v"])
