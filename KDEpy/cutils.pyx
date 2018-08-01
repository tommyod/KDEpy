# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:30:38 2018

@author: Tommy
"""

def iterate_data_weighted(double[:] transformed_data, double[:] weights, double[:] result):
    """
    Iterate over data points and weights and assign linear weights to nearest grid points.
    """
    cdef int length_data, length_result, integral
    cdef double data_point, weight, fractional, frac_times_weight
    length_data = transformed_data.shape[0]
    length_result = transformed_data.shape[0]

    for i in range(length_data):
        data_point, weight = transformed_data[i], weights[i]
        integral, fractional = int(data_point), (data_point) % 1
        frac_times_weight = fractional * weight  # Compute product once
        result[integral + 1] += frac_times_weight
        result[integral] += weight - frac_times_weight

    return result

def iterate_data(double[:] transformed_data, double[:] result):
    """
    Iterate over data points and assign linear weights to nearest grid points.
    """
    cdef int length_data, length_result, integral
    cdef double data_point, weight, fractional
    length_data = transformed_data.shape[0]
    length_result = transformed_data.shape[0]

    for i in range(length_data):
        data_point = transformed_data[i]
        integral, fractional = int(data_point), (data_point) % 1
        result[integral] += (1 - fractional)
        result[integral + 1] += fractional

    return result
