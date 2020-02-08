#! /usr/bin/env python3
"""
This module implements the rhs and compute error.
Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

import numpy as np

def u(v, k=1.0):
    """ Example function.
    Parameters:
    -----------
    v : ndarray
        One dimensional vector containing the grid points. Must not be a python list.
    k : float
        A constant, kappa.

    Returns:
    --------
    float
        The solution to the equation sum_{l = 1}^d x_l * sin(k * pi * x_l)
    """
    return np.prod([x * np.sin(k * np.pi * x) for x in v])

def f(v, k=1.0):
    """ Derivative of the example function.
    Parameters:
    -----------
    v : ndarray
        One dimensional vector containing the grid points. Must not be a python list.
    k : float
        A constant, kappa.

    Returns:
    --------
    float or None
        The solution to the derivative. If v has more than three entries, then the function will return None.
    """
    kpi = k * np.pi
    if v.size == 1:
        return kpi * (kpi * v[0] * np.sin(kpi * v[0]) - 2 * np.cos(kpi * v[0]))
    
    elif v.size == 2:
        sum1 = kpi * v[1] * np.sin(kpi * v[1]) * (kpi * v[0] * np.sin(kpi * v[0]) - 2 * np.cos(kpi * v[0]))
        sum2 = kpi * v[0] * np.sin(kpi * v[0]) * (kpi * v[1] * np.sin(kpi * v[1]) - 2 * np.cos(kpi * v[1]))
        return sum1 + sum2

    elif v.size == 3:
        x1, x2, x3 = v[0], v[1], v[2]
        kpi = k * np.pi

        sum1 = kpi * x2 * x3 * np.sin(kpi * x2) * np.sin(kpi * x3) * (kpi * x1 * np.sin(kpi * x1) - 2 * np.cos(kpi * x1))
        sum2 = kpi * x1 * x3 * np.sin(kpi * x1) * np.sin(kpi * x3) * (kpi * x2 * np.sin(kpi * x2) - 2 * np.cos(kpi * x2))
        sum3 = kpi * x1 * x2 * np.sin(kpi * x1) * np.sin(kpi * x2) * (kpi * x3 * np.sin(kpi * x3) - 2 * np.cos(kpi * x3))

        return sum1 + sum2 + sum3

    return None

def main():
    """
    As this module is purely there to seperate the example function, the main function does nothing.
    """
    pass

if __name__ == '__main__':
    main()