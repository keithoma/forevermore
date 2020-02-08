#! /usr/bin/env python3

import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy

import block_matrix
import functions
import linear_solvers
import rhs

matplotlib.use('TkAgg')

def solve_poisson_sor(d, n, f, x0=None, omega=1.5):
    A = block_matrix.BlockMatrix(d, n).data.tocsr()
    b = rhs.rhs(d, n, f)
    if x0 is None:
        x0 = np.array([0 for _ in range((n - 1) ** d)])
    return linear_solvers.solve_sor(A, b, x0, omega=omega)

def optimal_omega(d, n):
    C_jac = scipy.sparse.identity((n - 1) ** d) - (1 / (2 * d)) * block_matrix.BlockMatrix(d, n).data
    spectral = max(scipy.linalg.eigh(C_jac.todense())[0])
    return 1 + (spectral / (1 + math.sqrt(1 - spectral ** 2))) ** 2

def draw_optimal_omega(max_n=30):
    for d in [1, 2, 3]:
        omegas = [optimal_omega(d, n) for n in range(2, max_n)]
        plt.plot([n for n in range(2, max_n)], omegas, label="error for $d$ = {}".format(d), linewidth=3)

    plt.xlabel("Number of Grid Points, $N$", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error of $u$ and $\hat{u}$ depending on $N$", fontsize=24)
    plt.show()

def error_sor(d, n, f, u, omega=1.94):
    u_hat = solve_poisson_sor(d, n, f, omega=omega)

    grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]
    if d == 1:
        u_vec = np.array([u(np.array([x])) for x in grid])
    elif d == 2:
        u_vec = np.array([u(np.array([y, x])) for x in grid for y in grid])
    elif d == 3:
        u_vec = np.array([u(np.array([z, y, x])) for x in grid for y in grid for z in grid])
    else:
        pass

    print("Currently at sor: d = {}, n = {}".format(d, n))

    return np.linalg.norm(np.subtract(u_vec, u_hat), np.inf)

def draw_error_sor(f, u, dlist=[1, 2, 3], max_n=10):
    for d in dlist:
        errors = [error_sor(d, n, f, u) for n in range(2, max_n)]
        plt.loglog([n for n in range(2, max_n)], errors, label="error for $d$ = {}".format(d), linewidth=3)

    plt.xlabel("Number of Grid Points, $N$", fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.legend(fontsize=24)
    # pylint: disable=anomalous-backslash-in-string
    plt.title("Plot of error of $u$ and $\hat{u}$ depending on $N$", fontsize=24)
    plt.show()

def draw_error_both(f, u, dlist=[1, 2, 3], max_n=10):
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
    # print(error_sor(1, 10, functions.f, functions.u, omega=1.1))
    # print(error_sor(1, 10, functions.f, functions.u, omega=1.2))
    # print(error_sor(1, 10, functions.f, functions.u, omega=1.3))
    # print(error_sor(1, 10, functions.f, functions.u, omega=1.4))
    # print("-----")
    # print(error_sor(2, 10, functions.f, functions.u, omega=1.1))
    # print(error_sor(2, 10, functions.f, functions.u, omega=1.2))
    # print(error_sor(2, 10, functions.f, functions.u, omega=1.3))
    # print(error_sor(2, 10, functions.f, functions.u, omega=1.4))
    # print("-----")
    # print(error_sor(1, 5, functions.f, functions.u, omega=1.1))
    # print(error_sor(1, 5, functions.f, functions.u, omega=1.2))
    # print(error_sor(1, 5, functions.f, functions.u, omega=1.3))
    # print(error_sor(1, 5, functions.f, functions.u, omega=1.4))
    # print("-----")
    # print(error_sor(2, 5, functions.f, functions.u, omega=1.1))
    # print(error_sor(2, 5, functions.f, functions.u, omega=1.2))
    # print(error_sor(2, 5, functions.f, functions.u, omega=1.3))
    # print(error_sor(2, 5, functions.f, functions.u, omega=1.4))

    # print(error_sor(1, 15, functions.f, functions.u, omega=1.7))
    # print(error_sor(1, 15, functions.f, functions.u, omega=1.8))
    # print(error_sor(1, 15, functions.f, functions.u, omega=1.9))
    # print(error_sor(1, 15, functions.f, functions.u, omega=1.99))

    # draw_error_sor(functions.f, functions.u)
    draw_error_both(functions.f, functions.u, dlist=[1], max_n=20)



    # print(solve_poisson_sor(2, 5, functions.f))
    # print(error_sor(2, 5, functions.f, functions.u))
    # draw_error_sor(functions.f, functions.u, dlist=[1], max_n=25)
    # draw_error_both(functions.f, functions.u, dlist=[1], max_n=20)
    # optimal_omega(3, 15)
    # draw_optimal_omega(15)

if __name__ == '__main__':
    main()
