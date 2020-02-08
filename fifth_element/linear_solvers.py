#! /usr/bin/env python3
"""
"This module implements."
-- linear solving.


Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import random

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
    _ = slina.spsolve(sm.csc_matrix(slina.inv(sm.csc_matrix(pr))), b)
    _ = slina.spsolve_triangular(l, _, lower=True)
    _ = slina.spsolve_triangular(u, _, lower=False)
    _ = slina.spsolve(sm.csc_matrix(slina.inv(sm.csc_matrix(pc))), _)
    return _

def solve_sor(A, b, x0, params=dict(eps=1e-8, max_iter=1000, min_red=1e-4), omega=1.99):
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
        iterates of the algorithm. First entry is ‘x0‘.
    list
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., ‘eps=0‘ and ‘max_iter=0‘, etc.
    """
    def termination(x_k, it, last_residual):
        _ = np.matmul(A.toarray(), x_k)
        _ = np.subtract(_, b)
        residual = np.linalg.norm(_, np.inf)
        if residual < params["eps"]:
            return True
        elif it > params["max_iter"]:
            return True
        elif abs(last_residual - residual) < params["min_red"]:
            return True
        else:
            return False

    def next_x(x_k, it=0, last_residual=0):
        # print("x_k = {}".format(x_k))
        it += 1
        if termination(x_k, it, last_residual):
            return x_k
        else:
            sol_x = []
            for i in range(x0.size):
                sum1 = sum([A[i, j] * sol_x[j] for j in range(i)])
                sum2 = sum([A[i, j] * x_k[j] for j in range(i + 1, x0.size)])
                sol_x.append((1 - omega) * x_k[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2))
            
            _ = np.matmul(A.toarray(), x_k)
            _ = np.subtract(_, b)
            this_residual = np.linalg.norm(_, np.inf)
            return next_x(sol_x, it, this_residual)
    
    return next_x(x0)

def main():
    """
    JOIN THE GLORIOUS MAIN FUNCTION. Uses the test function to demonstrate.
    """
    mat = sm.csr_matrix(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))

    b = np.array([1, 2, 3, 4])
    x0 = np.array([0, 0, 0, 0])

    print(solve_sor(mat, b, x0))

if __name__ == '__main__':
    main()
