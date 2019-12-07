#! /usr/bin/env python3
"""
This module implements the rhs and compute error.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import numpy as np

import block_matrix
import linear_solvers

def rhs(d, n, f):
    """ Computes the right-hand side vector 'b' for a given function 'f'.
    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem
    Returns
    -------
    np.array or None
        Vector to the right-hand-side f. Returns None if d > 3.
    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """
    if d < 1 or n < 2:
        raise ValueError("We require d >= 1 and n >= 2!")

    grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]

    if d == 1:
        return (1 / n) ** 2 * np.array([f(x) for x in grid])
    elif d == 2:
        return (1 / n) ** 2 * np.array([f([y, x]) for x in grid for y in grid])
    elif d == 3:
        return (1 / n) ** 2 * np.array([f([z, y, x]) for x in grid for y in grid for z in grid])
    return None

def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the max-norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of ’numpy’
        Finite difference approximation of the solution of the Poisson problem
        at the disretization points
    u : callable
        Solution of the Possion problem
        The calling signature is ’u(x)’. Here ’x’ is a scalar
        or array_like of ’numpy’. The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the disretization points
    """
    grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]

    if d == 1:
        u_vec = np.array([u(x) for x in grid])
    elif d == 2:
        u_vec = np.array([u([y, x]) for x in grid for y in grid])
    elif d == 3:
        u_vec = np.array([u([z, y, x]) for x in grid for y in grid for z in grid])

    return np.linalg.norm([abs(ai - bi) for ai, bi in zip(u_vec, hat_u)], np.inf) if d >= 1 or d <= 3 else 0

def draw_error(max_n=15):
    pass

# DEBUG

def main():
    pass

if __name__ == '__main__':
    main()
