#! /usr/bin/env python3
"""
This module implements.

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

    # in the case d > 3
    _ = 0
    if d == 1:
        _ = np.linalg.norm([abs(ai - bi)
            for ai, bi in zip( np.array( [u(x) for x in grid] ), hat_u )], np.inf)
    elif d == 2:
        _ = np.linalg.norm([abs(ai - bi)
            for ai, bi in zip( np.array( [u([y, x]) for x in grid for y in grid] ), hat_u )], np.inf)
    elif d == 3:
        _ = np.linalg.norm([abs(ai - bi)
            for ai, bi in zip( np.array( [u([z, y, x]) for x in grid for y in grid for z in grid] ), hat_u )], np.inf)
    return _

def draw_error(max_n=15):
    pass

# DEBUG

def test_compute_error_1():
    def test_fun(x, k):
        _ = 1
        for l in range(0, x.size):
            _ = _ * x[l] * np.sin(k * np.pi * x[l])
        return _

    mat = block_matrix.BlockMatrix(3, 5)


    sol = test_fun(np.array([0.33, 0.33]), 5)
    print(sol)

def test_compute_error():
    def u_2(v):
        x, y = v[0], v[1]
        return x * np.sin(10*x*np.pi)*y*np.sin(10*y*np.pi)

    def thef(v):
        x, y = v[0], v[1]
        return (-1)*((-1)*20*np.pi*y*np.sin(10*np.pi*y)*(5*np.pi*x*np.sin(10*np.pi) - np.cos(10*np.pi*x)) + (-1)*20*np.pi*np.sin(10*np.pi*x) * (5*np.pi*y*np.sin(10*np.pi*y)-np.cos(10*np.pi*y)))

    n = 200

    b = rhs(2, n, thef)

    mat = block_matrix.BlockMatrix(2, n)
    u_hat = linear_solvers.solve_lu(*mat.get_lu(), b)
    print(b)
    print()
    print(u_hat)
    print()
    error = compute_error(2, n, u_hat, u_2)
    print(error)



def main():
    test_compute_error()

if __name__ == '__main__':
    main()
