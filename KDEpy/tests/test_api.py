#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from KDEpy import NaiveKDE
import itertools
import pytest


def test_1d_data_inputs():
    """
    Test that passing data as lists, tuples and np arrays is all okay.
    """
    
    input_data = [1, 10, 100]
    
    k = NaiveKDE(kernel='gaussian', bw=1)
    
    k.fit(np.array(input_data))
    x_1, y_1 = k.evaluate()
    
    k.fit(list(input_data))
    x_2, y_2 = k.evaluate()
    
    k.fit(tuple(input_data))
    x_3, y_3 = k.evaluate()
    
    k.fit(np.array(input_data).reshape(-1, 1))
    x_4, y_4 = k.evaluate()
    
    assert np.allclose(y_1, y_2)
    assert np.allclose(y_2, y_3)
    assert np.allclose(y_3, y_4)


def data_must_have_length():
    
    input_data = np.array([])
    k = NaiveKDE(kernel='gaussian', bw=1)
    
    with pytest.raises(ValueError):
        k.fit(np.array(input_data)) 
        
def grid_must_have_length():
    
    input_data = np.array([3, 4])
    k = NaiveKDE(kernel='gaussian', bw=1)
    
    with pytest.raises(ValueError):
        k.fit(np.array(input_data))
        k.evaluate(np.array([]))
        
        
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    