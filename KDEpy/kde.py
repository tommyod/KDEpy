#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np

if False:
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=['.', '--doctest-modules', '-v'])
    
    from KDEpy import NaiveKDE, TreeKDE
    from KDEpy.binning import linbin_numpy
    import matplotlib.pyplot as plt
    from time import perf_counter
    
    np.random.seed(123)
    data = np.random.lognormal(3, 0.2, 2**12)
    
    st = perf_counter()
    x, y = NaiveKDE('epa').fit(data).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='NaiveKDE')
    
    st = perf_counter()
    x, y = TreeKDE('epa').fit(data).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='TreeKDE')
    
    st = perf_counter()
    grid = np.linspace(np.min(data) - 5, np.max(data) + 5, 2**6)
    weights = linbin_numpy(data, grid)
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(grid, weights, '-o', label='linbin_numpy')
    
    st = perf_counter()
    x, y = TreeKDE('epa').fit(grid, weights=weights).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='TreeKDE on binned')
    
    plt.scatter(data, np.zeros_like(data))
    plt.legend(loc='best')
    
    
import numpy as np
from KDEpy.binning import linbin_numpy
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.optimize import fsolve, brentq, minimize

data = np.concatenate((np.random.randn(50)/10 + 1, np.random.randn(50)/10 + 2))
#data = np.array([1, 1, 1], dtype=np.float)
n = 2**12
MIN = 0
MAX = 3

# set up the grid over which the density estimate is computed;
R=MAX-MIN
dx=R/(n-1)
xmesh= MIN + np.linspace(MIN, MAX, n);
N = len(data)

# bin the data uniformly using the grid defined above;
initial_data = linbin_numpy(data, xmesh) / N
#initial_data = np.bincount(data, 
#                           minlength=len(xmesh) - 1)
initial_data = initial_data / np.sum(initial_data)

#plt.plot(xmesh, initial_data)

a = fftpack.dct(initial_data) # discrete cosine transform of initial data
#            N-1
# y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
#            n=0
# now compute the optimal bandwidth^2 using the referenced method
I = np.arange(1, n-1)**2
a2 = a[2:]**2 / 4

# use  fzero to solve the equation t=zeta*gamma^[5](t)
# t_star=root(@(t)fixed_point(t,N,I,a2),N);

def fixed_point(t, N, I, a2):
    """
    
    Examples
    >>> # From the matlab code
    >>> ans = fixed_point(0.01,50,np.arange(1, 51),np.arange(1, 51))
    >>> assert np.allclose(ans, 0.009947962622371)
    >>> # another
    >>> ans = fixed_point(0.07,25,np.arange(1, 11),np.arange(1, 11))
    >>> assert np.allclose(ans, 0.069100181315957)
    """
    
    # This is important, as the powers might overflow if not done
    I = np.asfarray(I)
    a2 = np.asfarray(a2)
    
    # l = 7 corresponds to the 5 steps recommended in the paper
    l = 7
    f = 2 * np.pi**(2 * l) * np.sum(np.power(I, l) * a2 * 
                                    np.exp(-I * np.pi**2 * t))

    assert f > 0
    for s in reversed(range(2, l)):
        #print(' ', s)
        odd_numbers_prod = np.product(np.arange(1, 2 * s + 1, 2, 
                                                dtype=np.float64))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1/2) ** (s + 1 / 2)) / 3
        time = np.power((2 * const * K0 / (N * f)), 
                        (2. / (3. + 2. * s)))
        f = 2 * np.pi**(2 * s) * np.sum(np.power(I, s) * a2 * 
                                        np.exp(-I * np.pi**2 * time))

        
        
    return t - ( 2 * N * np.sqrt(np.pi) * f)**(-2/5)

def root(function, N, args):
    """
    
    >>> # From the matlab code
    >>> ans = root(fixed_point, N=50, args=(50, np.arange(1, 51), np.arange(1, 51)))
    >>> assert np.allclose(ans, 5.203713947289470e-05)
    """
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N-50)/1000;
    #print(tol)
    
    # While a solution is not found, increase tolerance and try again
    found = 0
    while found == 0:
        x, infodict, found, mesg  = fsolve(function, tol, 
                                         args=args, 
                                         full_output=1)
        tol = tol * 2
        if tol > 0.1:
            return minimize(function, tol, args=args)
            
    return x

t_star = root(fixed_point, N, args= (N, I, a2))

print(t_star)

# smooth the discrete cosine transform of initial data using t_star
# a_t=a.*exp(-[0:n-1]'.^2*pi^2*t_star/2);
          
a_t = a * np.exp(-np.arange(n, dtype=np.float64)**2 * np.pi ** 2 * t_star /2)
# now apply the inverse discrete cosine transform
density = fftpack.idct(a_t) / (2 * R) # 2 here to normalize.. why?

plt.plot(xmesh, density)
plt.scatter(data, np.zeros_like(data), marker='|', color = 'red')
print(density)

# assert fixed_point(0.0001,50,1:50,1:50) 4.7963e-05
    

a = fixed_point(0.5, 50., np.arange(1, 51, dtype=np.float64), np.arange(1, 51, dtype=np.float64))
print('----')
print(a)

if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    import pytest
    pass
    # pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys', '-k root'])



















