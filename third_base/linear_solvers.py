#! /usr/bin/env python3
"""
This module implements.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

def solve_lu(pr, l, u, pc, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition pr * A * pc = l * u.
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
