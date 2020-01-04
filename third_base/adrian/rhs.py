"""This program calculates the right sided vector for finding a solution to the
poisson problem and has also functions to calculate the error one makes by cal-
culating on pc and not by hand.

Author:  hollerju (589259), salihiad (572395)
Date: 06.12.2019

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import linear_solvers
import block_matrix

def cartesian_product(*arrays):
    """
    Funktion, um das kartesische Produkt eines ndarrays zu berechnen.

    Parameter:
    ----------
    *arrays : (eine endliche Anzahl an) numpy.ndarray

    Return:
    -------
    arr.reshape(-1, la) : numpy.ndarray
            Ein ndarray, der das kartesische Produkt der Parameter-Arrays sym-
            bolisiert
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def rhs(d, n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.
    Parameters
    ----------
    d : int
    Dimension of the space.
    n : int
    Number of intervals in each dimension.
    f : callable
    Function right-hand-side of Poisson problem. The calling signature is
    `f(x)`. Here `x` is a scalar or array_like of `numpy`. The return value
    is a scalar.
    Returns
    -------
    numpy.ndarray
    Vector to the right-hand-side f.
    Raises
    ------
    ValueError
    If d < 1 or n < 2.
    """
    # Raise ValueError, falls d < 1 oder n < 2
    if d < 1 or n < 2:
        raise ValueError
    # Fall d = 1
    if d == 1:
        values = np.linspace(1/n, (n-1)/n, n-1)
        b = f(values)
    # Fall d = 2
    if d == 2:
        one_dim = np.linspace(1/n, (n-1)/n, n-1)
        values = cartesian_product(one_dim, one_dim)
        b = []
        elements = int(values.size/2)
        for i in range(elements):
            safe = values[i][0]
            values[i][0] = values[i][1]
            values[i][1] = safe
        for i in range(0, len(values)):
            b.append(f(values[i]))
    # Fall d = 3
    if d == 3:
        one_dim = np.linspace(1/n, (n-1)/n, n-1)
        b = []
        values = cartesian_product(one_dim, one_dim, one_dim)
        elements = int(values.size/3)
        for i in range(elements):
            safe = values[i][0]
            values[i][0] = values[i][2]
            values[i][2] = safe
        for i in range(0, len(values)):
            b.append(f(values[i]))
    return ((1/n)**2)*np.array(b)


def plot_points(d, n):
    """
    Calculates the discretization of the domain

    Parameter
    ---------
    d : int
    the dimension of space

    Return
    ------
    values : numpy.ndarray
    the discretization of the domain
    """
    if d < 1 or n < 2:
        raise ValueError

    one_dim = np.linspace(1/n, (n-1)/n, n-1)
    if d == 1:
        return one_dim
    if d == 2:
        values = cartesian_product(one_dim, one_dim)
        elements = int(values.size/2)
        for i in range(elements):
            safe = values[i][0]
            values[i][0] = values[i][1]
            values[i][1] = safe
        return values
    if d == 3:
        values = cartesian_product(one_dim, one_dim, one_dim)
        elements = int(values.size/3)
        for i in range(elements):
            safe = values[i][0]
            values[i][0] = values[i][2]
            values[i][2] = safe
        return values

def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the max-norm.
    Parameters
    ----------
    d : int
    Dimension of the space
    n : int
    Number of intersections in each dimension
    hat_u : array_like of 'numpy'
    Finite difference approximation of the solution of the Poisson problem
    at the disretization points
    u : callable
    Solution of the Possion problem
    The calling signature is 'u(x)'. Here 'x' is a scalar
    or array_like of 'numpy'. The return value is a scalar.
    Returns
    -------
    float
    maximal absolute error at the disretization points
    """
    values = plot_points(d, n)
    error_list = []
    if d == 1:
        for i in range(0, len(values)):
            error_list.append(abs(hat_u[i] - u(values[i])))
    if d == 2:
        for i in range(0, len(values)):
            error_list.append(abs(hat_u[i] - u(values[i])))
    if d == 3:
        for i in range(0, len(values)):
            error_list.append(abs(hat_u[i] - u(values[i])))
    return max(error_list)

def graph_error(d, high_N, f, u):
    """Plots the errors of the approximate solution of the Poisson Problem in
    dependency of N.

    d : int
    the dimension of the space
    high_N : int
    the highest N we want to plot the error for
    f : callable
    the f which is the negative of the laplace-operator
    u : callable
    Solution of the Possion problem
    """
    error_list = []
    plt.style.use("seaborn-white")
    plt.figure(figsize=(14, 7))
    plt.gcf().canvas.set_window_title("Graph des Fehlers \
in Abhängigkeit von N")
    plt.title("Fehlerplot in Abhängigkeit von N")
    axis = plt.gca()
    for n in range(3, high_N):
        matrix = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)
        hat_u = linear_solvers.solve_lu(*matrix.lu, b)
        error_list.append(compute_error(d, n, hat_u, u))
    axis.plot(range(3, high_N), error_list, label="Fehlerplot in Abhängigkeit\
von N für d = " + str(d))
    plt.xlabel("n-Werte")
    plt.ylabel("Fehler")
    plt.grid(True)
    #axis.set_xscale('log')
    #axis.set_yscale('log')
    plt.legend()
    plt.show()

def compute_error_list(d, high_N, f, u):
    """ Computes the errors of the approximate solution of the poisson-problem
    in dependency of n.

    Input
    -----
    d : int
    the dimension of the space
    high_N : int
    the highest N we want to plot the error for
    f : callable
    the f which is the negative of the laplace-operator
    u : callable
    Solution of the Possion problem

    Return
    ------
    error_list : list
    list of maximal errors between the approxmative solution of the poisson
    problem and the real solution.
    """
    error_list = []
    for n in range(3, high_N):
        matrix = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)
        hat_u = linear_solvers.solve_lu(*matrix.lu, b)
        error_list.append(compute_error(d, n, hat_u, u))
    return error_list

def plot_all_errors(values, list_1, list_2, list_3):
    """Plots error-functions for d=1,2,3 to compare them

    Input
    -----
    values : list
    our input values into the error function
    list_1 : list
    error-list for d=1
    list_2 : list
    error_list for d=2
    list_3 : list
    error_list for d=3
    """
    plt.style.use("seaborn-white")
    plt.figure(figsize=(14, 7))
    plt.gcf().canvas.set_window_title("Graph des absoluten Fehlers \
in Abhängigkeit von n")
    plt.title("Fehlerplot in Abhängigkeit von n")
    axis = plt.gca()
    axis.plot(values, list_1, label="Fehlerplot für d=1")
    axis.plot(values, list_2, label="Fehlerplot für d=2")
    axis.plot(values, list_3, label="Fehlerplot für d=3")
    plt.xlabel("n-Werte")
    plt.ylabel("Fehler")
    plt.grid(True)
    plt.legend()
    plt.show()
