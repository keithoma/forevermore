#! /usr/bin/env python3
""" This module verifies the results of the accompanying protocol.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import block_matrix
import functions
import linear_solvers
import rhs

matplotlib.use('TkAgg')

PRESET = dict(eps=1e-8, max_iter=1000, min_red=1e-4)
MAX_N = 10

def solve_poisson_sor(d, n, f, x0=None, params=PRESET, omega=1.5):
    """ Solves the poisson problem using the SOR algorithm.

    Params:
    -------
    d : int
        The dimension of the domain.
    n : int
        The number of grid points.
    f : callable
        The right hand side of the equation will be build from this function.
    x0 : numpy.ndarray, optional
        The initial guess for the algorithm.
    params : dict, optional
        A dictionary containing the termination conditions.
    omega : float
        The omega for the algorithm.

    Returns:
    --------
    numpy.ndarray
        The solution of the equation.
    """
    A = block_matrix.BlockMatrix(d, n).data.tocsr()
    b = rhs.rhs(d, n, f)
    if x0 is None:
        x0 = np.array([0 for _ in range((n - 1) ** d)])
    return linear_solvers.solve_sor(A, b, x0=x0, params=params, omega=omega)

def error_sor(d, n, f, u, x0=None, params=PRESET, omega=1.5):
    """ Computes the error between the solution from the SOR algorithm and the analytic one.

    Params:
    -------
    d : int
        The dimension of the domain.
    n : int
        The number of grid points.
    f : callable
        The right hand side of the equation will be build from this function.
    u : callable
        The analytic solution of x will be build from this function.
    x0 : numpy.ndarray, optional
        The initial guess for the algorithm.
    params : dict, optional
        A dictionary containing the termination conditions.
    omega : float
        The omega for the algorithm.

    Returns:
    --------
    float
        The error of the SOR solution. This is computed by using the maximum vector norm.
    """
    u_hat = solve_poisson_sor(d, n, f, x0=x0, params=params, omega=omega)

    grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]
    if d == 1:
        u_vec = np.array([u(np.array([x])) for x in grid])
    elif d == 2:
        u_vec = np.array([u(np.array([y, x])) for x in grid for y in grid])
    elif d == 3:
        u_vec = np.array([u(np.array([z, y, x])) for x in grid for y in grid for z in grid])
    else:
        pass

    return np.linalg.norm(np.subtract(u_vec, u_hat), np.inf)

def draw_error_sor(f, u, dlist=[1, 2, 3], max_n=MAX_N, params=PRESET):
    """ This function draws the error computed by the function above.

    Params:
    -------
    f : callable
        The right hand side of the equation will be build from this function.
    u : callable
        The analytic solution of x will be build from this function.
    dlist : list, optional
        A list containing dimension for which the plot is drawn. Especially useful if one has a
        damn slow computer which can't even run lol or any other game for that matter on any
        other graphic setting than the lowest, but thankfully, the author is getting a new one.
    max_n : int, optional
        The maximum number of n for which the plot is drawn.
    params : dict, optional
        The termination condition for the SOR algorithm.
    """
    for d in dlist:
        errors = [error_sor(d, n, f, u, params=params) for n in range(2, max_n)]
        plt.loglog([n for n in range(2, max_n)], errors,
                   label="error for $d$ = {}".format(d), linewidth=3)

    plt.xlabel("Number of Grid Points, $N$", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error of $u$ and $\hat{u}$ depending on $N$", fontsize=24)
    plt.show()

def draw_error_sor_modified(f, u, dlist=[1, 2, 3], max_n=MAX_N, epsk=1):
    """ This function is the modified version of the function above to verify the arguments made
    in the protocol.

    Params:
    -------
    f : callable
        The right hand side of the equation will be build from this function.
    u : callable
        The analytic solution of x will be build from this function.
    dlist : list, optional
        A list containing dimension for which the plot is drawn.
    max_n : int, optional
        The maximum number of n for which the plot is drawn.
    epsk : int, optional
        The exponent which alters the epsilon in the termination condition.
    """
    if epsk < 0:
        expo = 0
    elif epsk == 1:
        expo = -1
    else:
        expo = 2

    params = dict(eps=1e-8, max_iter=1000, min_red=1e-4)
    for d in dlist:
        errors = []
        for n in range(2, max_n):
            params["eps"] = (1 / (n + 1)) ** epsk
            errors.append(error_sor(d, n, f, u, params=params))
        plt.loglog([n for n in range(2, max_n)], errors,
                   label="error for $d$ = {}".format(d), linewidth=3)

    plt.loglog([x for x in range(2, max_n)], [1 / (x ** expo) for x in range(2, max_n)],
               label="$f(n) = 1 / n^{}$".format(expo), linestyle="--", linewidth=3)

    plt.xlabel("Number of Grid Points, $N$", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error $k = {}$".format(epsk), fontsize=24)
    plt.show()

def draw_error_both(f, u, dlist=[1, 2, 3], max_n=MAX_N):
    """ This function draws a plot to compare the SOR algorithm and the LU-decomposition.

    Params:
    -------
    f : callable
        The right hand side of the equation will be build from this function.
    u : callable
        The analytic solution of x will be build from this function.
    dlist : list, optional
        A list containing dimension for which the plot is drawn.
    max_n : int, optional
        The maximum number of n for which the plot is drawn.
    """
    for d in dlist:
        errors_sor = [error_sor(d, n, f, u) for n in range(2, max_n)]
        plt.loglog([n for n in range(2, max_n)], errors_sor,
                   label="error SOR for $d$ = {}".format(d), linewidth=3)

        errors_lu = []
        for n in range(2, max_n):
            hat_u = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                            rhs.rhs(d, n, functions.f))
            errors_lu.append(rhs.compute_error(d, n, hat_u, functions.u))

        plt.loglog([n for n in range(2, max_n)], errors_lu,
                   label="error LU for $d$ = {}".format(d), linewidth=3, linestyle="--")

    plt.xlabel("Number of Grid Points, $N$", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error of $u$ and $\hat{u}$ depending on $N$", fontsize=24)
    plt.show()

def main():
    """ The main function reproduces the results of the protocol.
    """
    print("SECTION: OPTIMAL OMEGA")
    print()
    print("d 1 | n 10 | omega 1.1  | Error = {}".format(error_sor(1, 10, functions.f, functions.u, omega=1.1)))
    print("d 1 | n 10 | omega 1.2  | Error = {}".format(error_sor(1, 10, functions.f, functions.u, omega=1.2)))
    print("d 1 | n 10 | omega 1.3  | Error = {}".format(error_sor(1, 10, functions.f, functions.u, omega=1.3)))
    print("d 1 | n 10 | omega 1.4  | Error = {}".format(error_sor(1, 10, functions.f, functions.u, omega=1.4)))
    print("----- ----- -----")
    print("d 2 | n 10 | omega 1.1  | Error = {}".format(error_sor(2, 10, functions.f, functions.u, omega=1.1)))
    print("d 2 | n 10 | omega 1.2  | Error = {}".format(error_sor(2, 10, functions.f, functions.u, omega=1.2)))
    print("d 2 | n 10 | omega 1.3  | Error = {}".format(error_sor(2, 10, functions.f, functions.u, omega=1.3)))
    print("d 2 | n 10 | omega 1.4  | Error = {}".format(error_sor(2, 10, functions.f, functions.u, omega=1.4)))
    print("----- ----- -----")
    print("d 1 | n 15 | omega 1.8  | Error = {}".format(error_sor(1, 15, functions.f, functions.u, omega=1.8)))
    print("d 1 | n 15 | omega 1.9  | Error = {}".format(error_sor(1, 15, functions.f, functions.u, omega=1.9)))
    print("d 1 | n 15 | omega 1.99 | Error = {}".format(error_sor(1, 15, functions.f, functions.u, omega=1.99)))
    print("----- ----- -----")
    print("\n\n\n")
    draw_error_sor(functions.f, functions.u)
    draw_error_both(functions.f, functions.u, dlist=[1], max_n=15)
    for k in [2, 1, 4, -2]:
        draw_error_sor_modified(functions.f, functions.u, max_n=8, epsk=k)

if __name__ == '__main__':
    main()
