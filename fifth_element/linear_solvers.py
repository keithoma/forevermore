#! /usr/bin/env python3
"""
"This module implements."
-- linear solving.


Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import numpy as np
from scipy import sparse as sm
from scipy.sparse import linalg as slina

def solve_lu(pr, l, u, pc, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition pr * A * pc = l * u.

    => A = pr^-1 * l * u * pc^-1

    Parameters
    ----------
    pr : scipy.sparse.csr_matrix
        row permutation matrix of LU-decomposition
    l : scipy.sparse.csr_matrix
        lower triangular unit diagonal matrix of LU-decomposition
    u : scipy.sparse.csr_matrix
        upper triangular matrix of LU-decomposition
    pc : scipy.sparse.csr_matrix
        column permutation matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """
    return slina.spsolve(
        sm.csc_matrix(
            slina.inv(sm.csc_matrix(pc))
        ),
        slina.spsolve_triangular(
            u,
            slina.spsolve_triangular(
                l,
                slina.spsolve(
                    sm.csc_matrix(slina.inv(sm.csc_matrix(pr))),
                    b
                ),
                lower=True),
            lower=False
        )
    )

def solve_sor(A, b, x0, params=dict(eps=1e-8, max_iter=1000, min_red=1e-4), omega=1.5):
    """ Solves the linear system Ax = b via the successive over relaxation method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray
        right-hand-side of the linear system
    x0 : numpy.ndarray
        initial guess of the solution
    params : dict, optional
        dictionary containing termination conditions
        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float
            minimal reduction of the residual in the infinity norm in every
            step. If set less or equal to 0 no constraint on the norm of the
            reduction of the residual is imposed.
    omega : float, optional
        relaxation parameter

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list
        iterates of the algorithm. First entry is 'x0'.
    list
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., 'eps=0' and 'max_iter=0', etc.
    """
    if params["max_iter"] <= 0 and params["eps"] <= 0 and params["min_red"] <= 0:
        raise ValueError("At least one of the termination conditions must be active!")

    def termination(list_of_x, list_of_residual):
        if len(list_of_x) > params["max_iter"] and params["max_iter"] > 0:
            return True, "max_iter"
        elif list_of_residual[-1] < params["eps"] and params["eps"] > 0:
            return True, "eps"
        elif len(list_of_residual) >= 2 and params["min_red"] > 0 and abs(list_of_residual[-2] - list_of_residual[-1]) < params["min_red"]:
            return True, "min_red"
        return False, None

    def next_x(x_k):
        """ Benutzt Summen-Formel """
        sol_x = []

        def fsum1(i):
            for j in range(i):
                yield A[i, j] * sol_x[j]
        #p1 = np.vectorize(lambda i, j: A[i, j] * sol_x[j], otypes=[float])

        def fsum2(i_plus_1, x_k_size):
            for j in range(i_plus_1, x_k_size):
                yield A[i, j] * x_k[j]
        #p2 = np.vectorize(lambda i, j: A[i, j] * x_k[j], otypes=[float])

        vsum = np.vectorize(lambda a, b: a - b, otypes=[float])
        for i in range(x_k.size):
            # diff = vsum(
            #     p1(i, range(i)),
            #     p2(i + 1, range(i + 1, x_k.size))
            # )
            sum1 = sum(fsum1(i))
            sum2 = sum(fsum2(i + 1, x_k.size))
            diff = sum1 - sum2

            sol_x.append((1 - omega) * x_k[i] + (omega / A[i, i]) * (b[i] - diff))
        return np.array(sol_x)

    def next_x2(x_k):
        """ Benutzt alternative Formel """
        L = sm.tril(A, k=-1)
        D = sm.diags(A.diagonal(), shape=A.shape, format="csr")
        U = sm.triu(A, k=1)

        return np.matmul(
            slina.inv(
                np.add(D, L.multiply(omega))
            ).toarray(),
            np.subtract(
                b * omega,
                np.matmul(
                    np.add(
                        U.multiply(omega),
                        D.multiply(omega - 1)
                    ).toarray(),
                    x_k
                )
            )
        )

    def residual(x_k):
        _ = A.toarray()
        _ = np.matmul(_, x_k)
        return np.linalg.norm(np.subtract(_, b), np.inf) # we can optimize here

    def construct():
        list_of_x, list_of_residual = [x0], [residual(x0)]
        while True:
            end, reason = termination(list_of_x, list_of_residual)
            if end:
                break
            list_of_x.append(next_x2(list_of_x[-1]))
            list_of_residual.append(residual(list_of_x[-1]))
        return reason, list_of_x, list_of_residual

    return construct()

def main():
    """
    JOIN THE GLORIOUS MAIN FUNCTION. Uses the test function to demonstrate.
    """
    mat = sm.csr_matrix(np.array([
        [ 4, -1, -6,  0],
        [-5, -4, 10,  8],
        [ 0,  9,  4, -2],
        [ 1,  0, -7,  5]
    ]))

    b = np.array([2, 21, -12, -6])
    x0 = np.array([0, 0, 0, 0])

    reason, list1, list2 = solve_sor(mat, b, x0, omega=0.5)

    print(reason)

    for item in list1:
        print(item)

if __name__ == '__main__':
    main()
