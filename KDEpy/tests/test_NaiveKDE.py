#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from KDEpy.NaiveKDE import NaiveKDE
import itertools
import pytest


args = itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2]], 
                         [1, 2, 3])


@pytest.mark.parametrize("data, split_index", args)
def test_additivity(data, split_index):
    """
    Test the additive propery of the KDE.
    """
    x = np.linspace(-10, 10)
    
    # Fit to add data
    y = NaiveKDE().fit(data).evaluate(x)
    
    # Fit to splits, and compensate for smaller data using weights
    weight_1 = split_index / len(data)
    y_1 = NaiveKDE().fit(data[:split_index]).evaluate(x) * weight_1
    
    weight_2 = (len(data) - split_index) / len(data)
    y_2 = NaiveKDE().fit(data[split_index:]).evaluate(x) * weight_2
    
    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)
    
    # import matplotlib.pyplot as plt
    
    # plt.plot(x, y, label='y')
    # plt.plot(x, y_1, label='y_1')
    # plt.plot(x, y_2, label='y_2')
    # plt.legend()
    # plt.show()
    
    
args = itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2]], 
                         [1, 2, 3])


@pytest.mark.parametrize("data, split_index", args)
def test_additivity_with_weights(data, split_index):
    """
    Test the additive propery of the KDE.
    """
    
    x = np.linspace(-10, 15)
    weights = np.arange(len(data)) + 1
    weights = weights / np.sum(weights)
    
    # Fit to add data
    y = NaiveKDE().fit(data, weights).evaluate(x)
    
    # Split up the data and the weights
    data = list(data)
    weights = list(weights)
    data_first_split = data[:split_index]
    data_second_split = data[split_index:]
    weights_first_split = weights[:split_index]
    weights_second_split = weights[split_index:]
    
    # Fit to splits, and compensate for smaller data using weights
    y_1 = (NaiveKDE().fit(data_first_split, weights_first_split)
           .evaluate(x) * sum(weights_first_split))
    
    y_2 = (NaiveKDE().fit(data_second_split, weights_second_split)
           .evaluate(x) * sum(weights_second_split))
    
    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)
 
        
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])