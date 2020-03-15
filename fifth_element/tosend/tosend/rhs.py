#! /usr/bin/env python3
"""
This module implements the rhs and compute error.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import numpy as np
from scipy.linalg import hilbert

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

import block_matrix # pylint: disable=wrong-import-position
import linear_solvers # pylint: disable=wrong-import-position
import functions # pylint: disable=wrong-import-position

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
        return (1 / n) ** 2 * np.array([f(np.array([x])) for x in grid])
    elif d == 2:
        return (1 / n) ** 2 * np.array([f(np.array([y, x])) for x in grid for y in grid])
    elif d == 3:
        return (1 / n) ** 2 * np.array([f(np.array([z, y, x])) for x in grid for y in grid for z in grid])
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
        u_vec = np.array([u([x]) for x in grid])
    elif d == 2:
        u_vec = np.array([u([y, x]) for x in grid for y in grid])
    elif d == 3:
        u_vec = np.array([u([z, y, x]) for x in grid for y in grid for z in grid])

    # pylint: disable=misplaced-comparison-constant
    return max([abs(ai - bi) for ai, bi in zip(u_vec, hat_u)]) if 1 <= d or d <= 3 else 0

def draw_error(max_n=15):
    """ This function draws the error for u and u_hat.

    Parameters
    ----------
    max_n : int
        The plot is drawn from 2 upto this int.
    """
    for d in [1, 2, 3]:
        errors = []
        for n in range(2, max_n):
            hat_u = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                            rhs(d, n, functions.f))
            errors.append(compute_error(d, n, hat_u, functions.u))

        plt.loglog([x for x in range(2, max_n)], errors, label="error for d = {}".format(d), linewidth=3)

    plt.xlabel("Number of Grid Points", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error of $u$ and $\hat{u}$ depending on n", fontsize=24)
    plt.show()

def draw_hilbert_cond(max_n=15):
    """ This function draws the condition for the Hilbert matrix.

    Parameters
    ----------
    max_n : int
        The plot is drawn from 3 upto this int.
    """
    condition = []
    for i in range(3, max_n):
        condition.append(np.linalg.cond(hilbert(i), np.inf))

    plt.loglog(range(3, max_n), [np.linalg.cond(hilbert(i), np.inf) for i in range(3, max_n)],
               label="Condition of the Hilbert Matrix", linewidth=3)
    axis = plt.gca()
    axis.set_yscale('log')

    plt.xlabel("Number of Rows/Columns of the Hilbert Matrix", fontsize=18)
    plt.ylabel("Condition", fontsize=18)
    plt.legend(fontsize=24)
    plt.show()

def main():
    """ A main function for demo.
    """
    d, n = 2, 4

    print("DEMONSTRATION OF MODULE")
    print("Consider sum_{l = 1}^d x_l * sin(k * pi * x_l).")
    print("We have d = {} and n = {}, then the right hand side of Ax = b would be:".format(d, n))
    print(np.array(rhs(d, n, functions.f)))
    print("And the error would be:")
    hat_u = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                    rhs(d, n, functions.f))
    print(compute_error(d, n, hat_u, functions.u))
    print()
    print("We can also the plot for the error.")
    print()
    print("See protocol for more information.")
    draw_error()
    draw_hilbert_cond()

if __name__ == '__main__':
    main()
