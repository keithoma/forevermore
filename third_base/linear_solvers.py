#! /usr/bin/env python3
"""
This module implements.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

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
    # remember kids, A = pr^-1 * l * u * pc^-1
    for item in [slina.inv(pc), u, l, slina.inv(pr)]:
        b = slina.spsolve(item, b)
    return b


def something():
    pr = scipy.sparse.csc_matrix(pr)
    pc = scipy.sparse.csc_matrix(pc)
    inverse_pr = sla.inv(pr)
    inverse_pc = sla.inv(pc)
    y_1 = sla.spsolve(inverse_pr, b)
    y_2 = sla.spsolve_triangular(l, y_1)
    y_3 = sla.spsolve_triangular(u, y_2, lower=False)
    x = sla.spsolve(inverse_pc, y_3)
    return x

def main():
    demo_matrix_1 = block_matrix.BlockMatrix(1,4)
    x = solve_lu(demo_matrix_1.get_lu(), [2,1,-4])
    print(x)
    print(demo_matrix_1.matrix.toarray())

if __name__ == '__main__':
    main()
