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
    Iterate over data points and weights and assign linear weights to nearest 
    grid points. This Cython implementation is for 1D data.
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
    This Cython implementation is for 1D data.
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
    Iterate over data points and weights and assign linear weights to nearest 
    grid points. This works, but is very slow and should not be used.
    TODO: Write a fast N dimensional linear binning loop in Cython.
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
            integrals_rev = list(integral for (integral, frac) 
                                 in reversed(cart_prod))
            
            index = int(sum((i * g**c) for ((c, i), g) in 
                            zip(enumerate(integrals_rev), grid_num)))
            
            # print(f'Placing {value} at index {index}, i.e. 
            # {grid_points[index % obs**dims,:]}')
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
    Iterate over data points and weights and assign linear weights to nearest 
    grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, weight, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y = data[i, 0], data[i, 1]
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
        index = y_integral + x_integral * grid_num[0]
        result[index % obs_tot] += (xy - x_fractional - y_fractional + 1) * weight
        
        # Bottom right
        index = y_integral + (x_integral + 1) * grid_num[0]
        result[index % obs_tot] += x_xy * weight
        
        # Top left
        index = (y_integral + 1) + x_integral * grid_num[0]
        result[index % obs_tot] += y_xy * weight
        
        # Top right
        index = (y_integral + 1) + (x_integral + 1) * grid_num[0]
        result[index % obs_tot] += xy * weight

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_2D(double[:, :] data, double[:] result, long[:] grid_num, 
                             int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest 
    grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y = data[i, 0], data[i, 1]
        
        x_integral = int(x)
        x_fractional = (x % 1)
        y_integral = int(y)
        y_fractional = (y % 1)

        # Computations with few flops
        xy = x_fractional * y_fractional
        y_xy = y_fractional - xy
        x_xy = x_fractional - xy
        
        # Bottom left
        index = y_integral + x_integral * grid_num[0]
        result[index % obs_tot] += (xy - x_fractional - y_fractional + 1)
        
        # Bottom right
        index = y_integral + (x_integral + 1) * grid_num[0]
        result[index % obs_tot] += x_xy
        
        # Top left
        index = (y_integral + 1) + x_integral * grid_num[0]
        result[index % obs_tot] += y_xy
        
        # Top right
        index = (y_integral + 1) + (x_integral + 1) * grid_num[0]
        result[index % obs_tot] += xy

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_weighted_3D(double[:, :] data, double[:] weights, 
                             double[:] result, long[:] grid_num, 
                             int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest 
    grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, z, weight, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y, z = data[i, 0], data[i, 1], data[i, 2]
        weight = weights[i]
        
        x_int = int(x)
        x_fractional = (x % 1)
        y_int = int(y)
        y_fractional = (y % 1)
        z_int = int(z)
        z_fractional = (z % 1)
        
        x, y, z = x_fractional, y_fractional, z_fractional
        
        # xyz (center)
        index = z_int + y_int * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * (1 - y) * (1 - z) * weight
        
        index = z_int + y_int * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * (1 - y) * (1 - z) * weight
        
        index = z_int + (y_int + 1) * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * y * (1 - z) * weight
        
        index = z_int + (y_int + 1) * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * y * (1 - z) * weight
        
        
        index = z_int + 1 + y_int * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * (1 - y) * z * weight
        
        index = z_int + 1 + y_int * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * (1 - y) * z * weight
        
        index = z_int + 1 + (y_int + 1) * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * y * z * weight
        
        index = z_int + 1 + (y_int + 1) * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * y * z * weight


    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_3D(double[:, :] data, double[:] result, long[:] grid_num, 
                    int obs_tot):
    """
    Iterate over data points and weights and assign linear weights to nearest 
    grid points.
    """
    cdef int length_data, index, i, x_integral, y_integral
    cdef double x, y, z, x_fractional, y_fractional, value
    
    data_length = data.shape[0]
    for i in range(data_length):
        x, y, z = data[i, 0], data[i, 1], data[i, 2]
        
        x_int = int(x)
        x_fractional = (x % 1)
        y_int = int(y)
        y_fractional = (y % 1)
        z_int = int(z)
        z_fractional = (z % 1)
        
        x, y, z = x_fractional, y_fractional, z_fractional
        
        # xyz (center)
        index = z_int + y_int * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * (1 - y) * (1 - z) 
        
        index = z_int + y_int * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * (1 - y) * (1 - z) 
        
        index = z_int + (y_int + 1) * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * y * (1 - z) 
        
        index = z_int + (y_int + 1) * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * y * (1 - z) 
        
        index = z_int + 1 + y_int * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * (1 - y) * z 
        
        index = z_int + 1 + y_int * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * (1 - y) * z 
        
        index = z_int + 1 + (y_int + 1) * grid_num[1] + x_int * grid_num[0]**2
        result[index % obs_tot] += (1 - x) * y * z 
        
        index = z_int + 1 + (y_int + 1) * grid_num[1] + (x_int + 1) * grid_num[0]**2
        result[index % obs_tot] += x * y * z 


    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_ND(double[:, :] data, double[:] result, long[:] grid_num, 
                    int obs_tot, long[:, :] binary_flgs):
    """
    Iterate over N-dimensional data and bin it.
    
    The idea behind this N-dimensional generalization is to pre-compute binary 
    numbers up to 2**dims. E.g. (0,0,0), (0,0,1), (0,1,0), (0,1,1), .. for 3 
    dimensions. Each tuple represent a corner in N-space. Let t_j be the tuple 
    binary value at index j, then the index computation may be expressed as
    
    index = SUM_{i=0} (int(x[n - j]) + 0**j) * grid_num[n - j]**j
    
    The first factor is either (x + 0) or (x + 1), depending on j.
    
    
    """
    cdef int obs, index, i, dims, corners, j, rev_j, flg, corner
    cdef int data_length, x_i_integer
    cdef double x, y, z, x_fractional, y_fractional, result_val, frac
    cdef double[:] x_i
    
    # Get the observations and dimensions of the data
    obs, dims = data.shape[0], data.shape[1]
    
    # For every dimension, there are two directions to find corners in
    corners = 2**dims

    # Loop through every data point
    for i in range(obs):
        
        # Retrieve the data point to consider
        x_i = data[i, :]
        
        # The data point will be 'assigned' to the 2**dims corners of the grid
        # that are closed to it. To do this, we loop through every corner
        for corner in range(corners):
            
            # For this corner, we must find the index of the `result` array
            # to input the computed result, and we must initialize the actual
            # result. Since index is computed additively and results as a
            # product, the initial values are the respective identities
            index = 0
            result_val = 1
            
            # To compute the index of this corner, and the value, we must
            # again loop through x_1, x_2, ..., d_x
            # The index is found by 
            # SUM_{i=0} (int(x[n - j]) + 0**flg) * grid_num[n - j]**flg
            # and the value is found by
            # PROD_{i=0} (1 - frac(x[n-1]))**flg * frac(x[n-1]) ** (1 - flg)

            for j in range(dims):
                # The reversed index, starting from the end: dims, ..., 2, 1, 0
                rev_j = dims - 1 - j
                # Get the flag indicating if we're considering (x) or (1-x)
                # in the product
                flg = binary_flgs[corner, j]
                # Moving this computation to an individual line speeds up the
                # Cython implementation by ~3 times. Worthwhile!
                x_i_integer = int(x_i[rev_j])
                index = index + (x_i_integer + 0**flg) * grid_num[rev_j] ** j
            
                # Compute part of the product, usings binary flags to indicate
                # (x) or (x-1)
                frac = x_i[rev_j] % 1
                result_val = result_val * (1 - frac)**flg * frac**(1 - flg)
            
            # Finished computing index and result, add to the grid corner point
            result[index % obs_tot] += result_val
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_data_ND_weighted(double[:, :] data, double[:] weights, double[:] result, 
                    long[:] grid_num, int obs_tot, long[:, :] binary_flgs):
    """
    Iterate over N-dimensional data and bin it.
    
    The idea behind this N-dimensional generalization is to pre-compute binary 
    numbers up to 2**dims. E.g. (0,0,0), (0,0,1), (0,1,0), (0,1,1), .. for 3 
    dimensions. Each tuple represent a corner in N-space. Let t_j be the tuple 
    binary value at index j, then the index computation may be expressed as
    
    index = SUM_{i=0} (int(x[n - j]) + 0**j) * grid_num[n - j]**j
    
    The first factor is either (x + 0) or (x + 1), depending on j.
    
    
    """
    cdef int obs, index, i, dims, corners, j, rev_j, flg, corner
    cdef int data_length, x_i_integer
    cdef double x, y, z, x_fractional, y_fractional, result_val, frac, weight
    cdef double[:] x_i
    
    obs, dims = data.shape[0], data.shape[1]
    corners = 2**dims

    for i in range(obs):
        x_i = data[i, :]
        weight = weights[i]

        for corner in range(corners):
            index = 0
            result_val = 1

            for j in range(dims):
                rev_j = dims - 1 - j
                flg = binary_flgs[corner, j]
                x_i_integer = int(x_i[rev_j])
                index = index + (x_i_integer + 0**flg) * grid_num[rev_j] ** j
                frac = x_i[rev_j] % 1
                result_val = result_val * (1 - frac)**flg * frac**(1 - flg)
            result[index % obs_tot] += result_val * weight
    
    return result