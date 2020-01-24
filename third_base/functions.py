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
    v : list
        The vector containing the grid points. Must not be np.array.
    k : float
        A constant.

    Returns:
    float
        The solution to the equation sum_{l = 1}^d x_l * sin(k * pi * x_l)
    """
    _ = 1
    vector_size = len(v) if isinstance(v, list) else 1
    for l in range(vector_size):
        _ = _ * v[l] * np.sin(k * np.pi * v[l])
    return _

def f(v, k=1.0):
    """ Derivative of the example function.

    Parameters:
    v : list
        The vector containing the grid points. Must not be np.array.
    k : float
        A constant.

    Returns:
    float
        The solution to the derivative
    """
    if len(v) == 1:
        return k * np.pi * (k * np.pi * v[0] * np.sin(k * np. pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))

    elif len(v) == 2:
        sum1 = k * np.pi * v[1] * np.sin(k * np.pi * v[1]) * \
               (k * np.pi * v[0] * np.sin(k * np.pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))
        sum2 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * \
               (k * np.pi * v[1] * np.sin(k * np.pi * v[1]) - 2 * np.cos(k * np.pi * v[1]))
        return sum1 + sum2

    elif len(v) == 3:
        x1, x2, x3 = v[0], v[1], v[2]
        kpi = k * np.pi

        sum1 = kpi * x2 * x3 * np.sin(kpi * x2) * np.sin(kpi * x3) * (2 * np.cos(kpi * x1) - kpi * x1 * np.sin(kpi * x1))
        sum2 = kpi * x1 * x3 * np.sin(kpi * x1) * np.sin(kpi * x3) * (2 * np.cos(kpi * x2) - kpi * x2 * np.sin(kpi * x2))
        sum3 = kpi * x1 * x2 * np.sin(kpi * x1) * np.sin(kpi * x2) * (2 * np.cos(kpi * x3) - kpi * x3 * np.sin(kpi * x3))

        return -(sum1 + sum2 + sum3)

    return None

def main():
    """
    As this module is purely there to seperate the example function, the main function does nothing.
    """
    pass

if __name__ == '__main__':
    main()
