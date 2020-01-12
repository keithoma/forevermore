"""
Program to solve linear systems with the LU-Method

Author:  hollerju (589259), salihiad (572395)
Date: 06.12.2019

"""


import scipy.sparse
from scipy.sparse import linalg as sla


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
    pr = scipy.sparse.csc_matrix(pr)
    pc = scipy.sparse.csc_matrix(pc)
    inverse_pr = sla.inv(pr)
    inverse_pc = sla.inv(pc)
    y_1 = sla.spsolve(inverse_pr, b)
    y_2 = sla.spsolve_triangular(l, y_1)
    y_3 = sla.spsolve_triangular(u, y_2, lower=False)
    solution = sla.spsolve(inverse_pc, y_3)
    return solution
