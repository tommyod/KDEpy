# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:30:38 2018

@author: Tommy
"""

cimport cython
import numpy as np
import itertools

# boundscheck(False) -> Cython is free to assume that indexing will not cause 
# any IndexErrors to be raised.

# wraparound(False) ->  If set to False, Cython is allowed to neither check 
# for nor correctly handle negative indices

# cdivision(True) -> If set to False, Cython will adjust the remainder and 
# quotient operators C types to match those of Python ints (which differ 
# when the operands have opposite signs) and raise a ZeroDivisionError 
# when the right operand is 0
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_weighted_N(double[:, :] data, double[:] weights, double[:] result, double[:] grid_num, int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest grid points.
    """
    cdef int length_data, index
    cdef double weight, value
    cdef double[:] observation
    length_data = data.shape[0]
    
    for i in range(length_data):
        observation = data[i, :]
        weight = weights[i]

        int_frac = list(((int(coordinate), 1 - (coordinate % 1)), 
                     (int(coordinate) + 1,  (coordinate % 1)))
                    for coordinate in observation)

        for cart_prod in itertools.product(*int_frac):
            fractions = [frac for (integral, frac) in cart_prod]
            
            # In reversed order
            integrals_rev = list(integral for (integral, frac) in reversed(cart_prod))
            
            index = int(sum((i * g**c) for ((c, i), g) in zip(enumerate(integrals_rev), grid_num)))
            

            # print(f'Placing {value} at index {index}, i.e. {grid_points[index % obs**dims,:]}')
            value = np.prod(fractions)
            result[int(index % obs_tot)] += value * weight
            
    return result
