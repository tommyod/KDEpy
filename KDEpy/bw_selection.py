#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.
"""
import numpy as np


def CV(data):
    """
    """
    pass


def scotts_rule(data):
    """
    
    """
    pass


def silvermans_rule(data):
    """
    """
    if len(data) == 1:
        return 1
    if len(data) < 1:
        raise ValueError('Data must be of length > 0.')
        
    sigma = np.std(data, ddof=1)
    # scipy.norm.ppf(.75) - scipy.norm.ppf(.25) -> 1.3489795003921634
    IQR = ((np.percentile(data, q=75) - np.percentile(data, q=25)) / 
           1.3489795003921634)

    sigma = min(sigma, IQR)
    return sigma * (np.size(data) * 3 / 4.) ** (-1 / 5)

    
_bw_methods = {'silverman': silvermans_rule}