#! /usr/bin/env python3
"""
This module implements the rhs and compute error.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

import block_matrix # pylint: disable=wrong-import-position
import linear_solvers # pylint: disable=wrong-import-position

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
        return (1 / n) ** 2 * np.array([f([x]) for x in grid])
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
        u_vec = np.array([u([x]) for x in grid])
    elif d == 2:
        u_vec = np.array([u([y, x]) for x in grid for y in grid])
    elif d == 3:
        u_vec = np.array([u([z, y, x]) for x in grid for y in grid for z in grid])

    # pylint: disable=misplaced-comparison-constant
    return np.linalg.norm([abs(ai - bi) for ai, bi in zip(u_vec, hat_u)], np.inf) if 1 <= d or d <= 3 else 0

def draw_error(max_n=15):
    """ This function draws the error for u and u_hat.

    Parameters
    ----------
    max_n : int
        The plot is drawn from 2 upto this int.
    """
    def example_function(v, k=1.0):
        """ Example function to draw an example graph.

        Parameters:
        v : list
            The vector containing the grid points.
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

    def example_derivative(v, k=1.0):
        """ Example function to draw an example graph.

        Parameters:
        v : list
            The vector containing the grid points.
        k : float
            A constant.

        Returns:
        float
            The solution to the differential of sum_{l = 1}^d x_l * sin(k * pi * x_l)
        """
        def partial(v, k, l):
            """ Helper function to make my life easier. Summand of the differential.

            Parameters
            ----------
            v : list
                The vector.
            k : float
                A constant.
            l : int
                The index of the current summand.

            Returns:
            float
                The solution for the partial differentiation with respect to x_l.
            """
            _ = k * np.pi * v[l] * np.sin(k * np.pi * v[l]) - 2 * np.cos(k * np.pi * v[l])
            for i in [d for d in range(0, len(v)) if d != l]: # skip over the variable to differentiate
                _ = _ * k * np.pi * v[i] * np.sin(k * np.pi * v[i])
            return _

        _ = 0

        for l in range(len(v)):
            _ = _ + partial(v, k, l)

        return _

    for d in [1, 2, 3]:
        errors = []
        for n in range(2, max_n):
            hat_u = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                            rhs(d, n, example_derivative))
            errors.append(compute_error(d, n, hat_u, example_function))

        plt.plot([x for x in range(2, max_n)], errors, label="error for d = {}".format(d))

    plt.xlabel('$n$')
    plt.ylabel('$error$')
    plt.legend()
    plt.title("Plot of error of u and u_hat depending on n")
    plt.show()

def main():
    """ A main function for demo.
    """
    def example_function(v, k=1.0):
        """ Example function to draw an example graph.

        Parameters:
        v : list
            The vector containing the grid points.
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

    def example_derivative(v, k=1.0):
        """ Example function to draw an example graph.

        Parameters:
        v : list
            The vector containing the grid points.
        k : float
            A constant.

        Returns:
        float
            The solution to the differential of sum_{l = 1}^d x_l * sin(k * pi * x_l)
        """
        def partial(v, k, l):
            """ Helper function to make my life easier. Summand of the differential.
            """
            _ = k * np.pi * v[l] * np.sin(k * np.pi * v[l]) - 2 * np.cos(k * np.pi * v[l])
            for i in [d for d in range(0, len(v)) if d != l]: # skip over the variable to differentiate
                _ = _ * k * np.pi * v[i] * np.sin(k * np.pi * v[i])
            return _

        _ = 0

        for l in range(len(v)):
            _ = _ + partial(v, k, l)

        return _

    d, n = 2, 4

    print("DEMONSTRATION OF MODULE")
    print("Consider sum_{l = 1}^d x_l * sin(k * pi * x_l).")
    print("We have d = {} and n = {}, then the right hand side of Ax = b would be:".format(d, n))
    print(np.array(rhs(d, n, example_function)))
    print("And the error would be:")
    hat_u = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                    rhs(d, n, example_function))
    print(compute_error(d, n, hat_u, example_derivative))
    print()
    print("We can also the plot for the error.")
    print()
    print("See protocol for more information.")
    draw_error(20)

if __name__ == '__main__':
    main()
