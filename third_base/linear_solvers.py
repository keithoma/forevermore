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
    _ = slina.spsolve(sm.csc_matrix(slina.inv(sm.csc_matrix(pr))), b)
    _ = slina.spsolve_triangular(l, _, lower=True)
    _ = slina.spsolve_triangular(u, _, lower=False)
    _ = slina.spsolve(sm.csc_matrix(slina.inv(sm.csc_matrix(pc))), _)
    return _

def test_validity(n_upto=20, num_test=20, show=False):
    """
    CHECKS THE ABOVE FUNCTION FOR CORECTNESS. Loops through some random values for b and compares the
    function above with the spsolve from SciPy.

    Parameters
    ----------
    n_upto : int
        Checks the correctness from 2 to this integer.
    num_test : int
        Number of tests for each n.
    show : boolean
        Option to print out the values.

    Returns
    -------
    boolean
        True if no wrongness was encountered, False if an error was seen.
    """
    for d in [1, 2, 3]:
        for n in range(2, n_upto):
            for i in range(0, num_test):
                mat = block_matrix.BlockMatrix(d, n)
                b = np.array([random.randint(-20, 20) for _ in range(0, mat.extend)])
                x_demo = solve_lu(*mat.get_lu(), b)
                x_true = slina.spsolve(sm.csc_matrix(mat.data), b)

                if np.all((x_demo, x_true)):
                    print("d = {} | n = {} | TEST {}/{} PASSED!".format(d, n, i + 1, num_test))
                    if show:
                        print()
                        print("A = {}".format(mat.data.toarray()))
                        print("b = {}\n".format(b))
                        print("x = {}".format(x_demo))
                        print()
                        print("============================================================\n")
                else:
                    print()
                    print("d = {} | n = {} | TEST {}/{} FAILED FOR".format(d, n, i + 1, num_test))
                    print("A = {}".format(mat.data.toarray()))
                    print("b = {}\n".format(b))
                    print("x = {}".format(x_demo))
                    print()
                    return False
    return True

def main():
    """
    JOIN THE GLORIOUS MAIN FUNCTION. Uses the test function to demonstrate.
    """
    print("WE SOLVE LINEAR EQUATION, Ax = b WHERE A AND b ARE KNOWN\n\n")
    test_validity(5, 1, True)

if __name__ == '__main__':
    main()
