#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from KDEpy.NaiveKDE import NaiveKDE
import pytest

kernel_funcs = list(NaiveKDE._available_kernels.values())


class TestNaiveKDE():
    
    @pytest.mark.parametrize("kernel, bw, n, expected_result", 
                             [('box', 0.1, 5, np.array([2.101278e-19, 
                                                        3.469447e-18, 
                                                        1.924501e+00, 
                                                        0.000000e+00, 
                                                        9.622504e-01])),
                              ('box', 0.2, 5, np.array([3.854941e-18, 
                                                        2.929755e-17,
                                                        9.622504e-01,
                                                        0.000000e+00,
                                                        4.811252e-01])),
                              ('box', 0.6, 3, np.array([0.1603751,
                                                        0.4811252,
                                                        0.4811252])),
                              ('tri', 0.6, 3, np.array([0.1298519,
                                                        0.5098009,
                                                        0.3865535])),
                              ('epa', 0.1, 6, np.array([0.000000e+00,
                                                        7.285839e-17,
                                                        2.251871e-01,
                                                        1.119926e+00,
                                                        0.000000e+00,
                                                        1.118034e+00])),
                              ('biweight', 2, 5, np.array([0.1524078,
                                                           0.1655184,
                                                           0.1729870,
                                                           0.1743973,
                                                           0.1696706]))])
    def test_against_R_density(self, kernel, bw, n, expected_result):
        """
        Test against the following function call in R:
            
            d <- density(c(0, 0.1, 1), kernel="{kernel}", bw={bw}, 
            n={n}, from=-1, to=1);
            d$y
        """
        data = np.array([0, 0.1, 1])
        x = np.linspace(-1, 1, num=n)
        y = NaiveKDE(kernel, bw=bw).fit(data).evaluate(x)
        assert np.allclose(y, expected_result, atol=10**(-2.7))

    @pytest.mark.parametrize("bw, n, expected_result", 
                             [(1, 3, np.array([0.17127129, 
                                               0.34595518, 
                                               0.30233275])),
                              (0.1, 5, np.array([2.56493684e-22, 
                                                 4.97598466e-06, 
                                                 2.13637668e+00, 
                                                 4.56012216e-04,
                                                 1.32980760e+00])),
                              (0.01, 3, np.array([0., 
                                                  13.29807601, 
                                                  13.29807601]))])
    def test_against_scipy_density(self, bw, n, expected_result):
        """
        Test against the following function call in SciPy:
            
            data = np.array([0, 0.1, 1])
            x = np.linspace(-1, 1, {n})
            bw = {bw}/np.asarray(data).std(ddof=1)
            density_estimate = gaussian_kde(dataset = data, bw_method = bw)
            y = density_estimate.evaluate(x)
            
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        """
        data = np.array([0, 0.1, 1])
        x = np.linspace(-1, 1, num=n)
        y = NaiveKDE(kernel='gaussian', bw=bw).fit(data).evaluate(x)
        assert np.allclose(y, expected_result)
        
        
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    