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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_weighted_2D(double[:, :] data, double[:] weights, 
                             double[:] result, long[:] grid_num, 
                             int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, weight, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y = data[i,0], data[i,1]
        weight = weights[i]
        
        x_integral = int(x)
        x_fractional = (x % 1)
        y_integral = int(y)
        y_fractional = (y % 1)
        
        #  | ---------------------------------
        #  |                   |              |
        #  |-------------------X--------------|
        #  |                   |              |
        #  |                   |              |
        #  |                   |              |
        #  |                   |              |
        #  | ---------------------------------

        # Computations with few flops
        xy = x_fractional * y_fractional
        y_xy = y_fractional - xy
        x_xy = x_fractional - xy
        
        # Bottom left
        index = y_integral + x_integral * grid_num[1]
        result[index % obs_tot] += (xy - x_fractional - y_fractional + 1) * weight
        
        # Bottom right
        index = y_integral + (x_integral + 1) * grid_num[1]
        result[index % obs_tot] += x_xy * weight
        
        # Top left
        index = (y_integral + 1) + x_integral * grid_num[1]
        result[index % obs_tot] += y_xy * weight
        
        # Top right
        index = (y_integral + 1) + (x_integral + 1) * grid_num[1]
        result[index % obs_tot] += xy * weight
        
#        # Bottom left
#        index = y_integral + x_integral * grid_num[1]
#        value = (1 - x_fractional) * (1 - y_fractional)
#        result[index % obs_tot] += value * weight
#        
#        # Bottom right
#        index = y_integral + (x_integral + 1) * grid_num[1]
#        value = (x_fractional) * (1 - y_fractional)
#        result[index % obs_tot] += value * weight
#        
#        # Top left
#        index = (y_integral + 1) + x_integral * grid_num[1]
#        value = (1 - x_fractional) * (y_fractional)
#        result[index % obs_tot] += value * weight
#        
#        # Top right
#        index = (y_integral + 1) + (x_integral + 1) * grid_num[1]
#        value = (x_fractional) * (y_fractional)
#        result[index % obs_tot] += value * weight

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_2D(double[:, :] data, double[:] result, long[:] grid_num, 
                             int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y = data[i,0], data[i,1]
        
        x_integral = int(x)
        x_fractional = (x % 1)
        y_integral = int(y)
        y_fractional = (y % 1)
        
        #  | ---------------------------------
        #  |                   |              |
        #  |-------------------X--------------|
        #  |                   |              |
        #  |                   |              |
        #  |                   |              |
        #  |                   |              |
        #  | ---------------------------------

        # Computations with few flops
        xy = x_fractional * y_fractional
        y_xy = y_fractional - xy
        x_xy = x_fractional - xy
        
        # Bottom left
        index = y_integral + x_integral * grid_num[1]
        result[index % obs_tot] += (xy - x_fractional - y_fractional + 1)
        
        # Bottom right
        index = y_integral + (x_integral + 1) * grid_num[1]
        result[index % obs_tot] += x_xy
        
        # Top left
        index = (y_integral + 1) + x_integral * grid_num[1]
        result[index % obs_tot] += y_xy
        
        # Top right
        index = (y_integral + 1) + (x_integral + 1) * grid_num[1]
        result[index % obs_tot] += xy

    return result