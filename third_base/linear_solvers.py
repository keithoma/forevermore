#! /usr/bin/env python3
"""
This module implements.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""
import numpy as np
from scipy import sparse as sm
from scipy.sparse import linalg as slina

import random

import block_matrix

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
    _ = slina.spsolve(slina.inv(sm.csc_matrix(pr)), b)
    _ = slina.spsolve_triangular(l, _, lower=True)
    _ = slina.spsolve_triangular(u, _, lower=False)
    _ = slina.spsolve(slina.inv(sm.csc_matrix(pc)), _)
    return _

# DEBUG FUNCTIONS FOR FUN

def test_validity(n_upto=20, num_test=20, show=False):
    for d in [1, 2, 3]:
        for n in range(2, n_upto):
            for i in range(0, num_test):
                mat = block_matrix.BlockMatrix(d, n)
                b = [random.randint(-20, 20) for _ in range(0, mat.extend)]
                x_demo = solve_lu(*mat.get_lu(), b)
                x_true = slina.spsolve(mat.data, b)

                if np.all((x_demo, x_true)):
                    print("d = {} | n = {} | TEST {}/{} PASSED!".format(d, n, i + 1, num_test))
                    if show:
                        print()
                        print("A = {}".format(mat.data.toarray()))
                        print("b = {}\n".format(b))
                        print("x = {}".format(x_demo))
                        print("==============================\n")
                else:
                    print("d = {} | n = {} | TEST {}/{} FAILED FOR".format(d, n, i + 1, num_test))
                    print("A = {}".format(mat.data.toarray()))
                    print("b = {}\n".format(b))
                    print("x = {}".format(x_demo))
                    return False
    return True

def test_speed(d=3, n=50, num_test=50):
    mat = block_matrix.BlockMatrix(d, n)
    for i in range(0, num_test):
        b = [random.randint(-20, 20) for _ in range(0, mat.extend)]
        x = solve_lu(*mat.get_lu(), b)
        print("SOLVED {}/{}".format(i + 1, num_test))

def main():
    test_validity(5, 1, True)

if __name__ == '__main__':
    main()
