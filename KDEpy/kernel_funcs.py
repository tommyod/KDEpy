#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:52:43 2018

@author: tommy
"""

import numpy as np
import collections.abc


def epanechnikov(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = 0.75 * (1 - x * x)[mask]
    return out


def gaussian(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def box(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = 0.5
    return out


def tri(x):
    out = np.zeros_like(x)
    out[x >= 0] = np.maximum(0, 1 - x)[x >= 0]
    out[x < 0] = np.maximum(0, 1 + x)[x < 0]
    return out


class Kernel(collections.abc.Callable):
    
    def __init__(self, function, expected_value=0, left_bw=1, right_bw=1):
        """
        Initialize a new kernel function.
        
        function: callable, numpy.arr -> numpy.arr
        expected_value : peak, typically 0
        left_bw: support to the left
        left_bw: support to the right
        """
        self.function = function
        self.expected_value = expected_value
        self.left_bw = left_bw
        self.right_bw = right_bw
    
    def evaluate(self, x, bw=1):
        """
        Evaluate the kernel.
        """
        real_bw = (bw / (self.left_bw + self.right_bw))
        return self.function(x / real_bw) / real_bw
    
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)
    
    
gaussian = Kernel(gaussian, 0, 3, 3)
box = Kernel(box, 0, 1, 1)
tri = Kernel(tri, 0, 1, 1)
epa = Kernel(epanechnikov, 0, 1, 1)

_kernel_functions = {'gaussian': gaussian,
                     'box': box,
                     'tri': tri,
                     'epa': epa}