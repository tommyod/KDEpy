#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API tests. Since BaseKDE is an abstract class, the testing is done using the
naiveKDE class instead. The tests here are related to input types.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from KDEpy.NaiveKDE import NaiveKDE
import itertools
import pytest


args = itertools.product([0.5, 1, 3], NaiveKDE._available_kernels)


@pytest.mark.parametrize("bw, kernel", args)
def test_1d_data_inputs(bw, kernel):
    """
    Test that passing data as lists, tuples and NumPy arrays are all ok.
    """
    input_data = [1, 2, 5, 10]

    k = NaiveKDE(kernel=kernel, bw=bw)
    # Arrays
    k.fit(np.array(input_data))
    x_1, y_1 = k.evaluate()

    # Lists
    k.fit(list(input_data))
    x_2, y_2 = k.evaluate()

    # Tuples
    k.fit(tuple(input_data))
    x_3, y_3 = k.evaluate()

    # Arrays of shape (obs, dims)
    k.fit(np.array(input_data).reshape(-1, 1))
    x_4, y_4 = k.evaluate()

    assert np.allclose(y_1, y_2)
    assert np.allclose(y_2, y_3)
    assert np.allclose(y_3, y_4)


def test_common_API_patterns():
    """
    Test common API patterns.
    """
    # Simplest way, with auto grid
    data = [1, 2, 5, 10]
    x, y = NaiveKDE().fit(data).evaluate()

    # Using a pre-defined grid
    x = np.linspace(-10, 50)
    y1 = NaiveKDE().fit(data).evaluate(x)

    # No chaining
    k = NaiveKDE()
    k.fit(data)
    y2 = k.evaluate(x)

    assert np.allclose(y1, y2)


def test_weights():
    """
    Test that the default weights are set to uniform.
    """
    data = [1, 2, 5, 10]
    x1, y1 = NaiveKDE().fit(data).evaluate()

    weights = np.array(np.ones_like(data)) / len(data)

    x2, y2 = NaiveKDE().fit(data, weights=weights).evaluate()

    assert np.allclose(y1, y2)


def test_data_must_have_length():
    """
    Test that an error is raised when the data has no length.
    """

    input_data = np.array([])
    k = NaiveKDE(kernel="gaussian", bw=1)

    with pytest.raises(ValueError):
        k.fit(np.array(input_data))


def test_grid_must_have_length():
    """
    Test that an error is raised when the grid has no length.
    """

    input_data = np.array([3, 4])
    k = NaiveKDE(kernel="gaussian", bw=1)

    with pytest.raises(ValueError):
        k.fit(np.array(input_data))
        k.evaluate(np.array([]))


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v"])
