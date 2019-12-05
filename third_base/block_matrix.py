#! /usr/bin/env python3
"""
This module implements the BlockMatrix class which is used to solve the Poisson problem.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

import numpy as np
from scipy import sparse as sm
from scipy.sparse import csc_matrix, linalg as sla
# from scipy import linalg

def construct(d, n):
    """
        Constructs block matrices arising from finite difference approximations of the Laplace
        operator.

        Parameters
        ----------
        d : int
            Dimension of the space
        n : int
            Number of intervals in each dimension

        Returns
        -------
        scipy.sparse.coo_matrix
            block_matrix in a sparse data format
    """
    def generate(l):
        res = [[{
            0: lambda: generate(l - 1),
            1: lambda: -1 * sm.identity((n - 1) ** (l - 1))
        }.get(abs(i - j), lambda: None)() for j in range(1, n)] for i in range(1, n)] if l > 1 else sm.coo_matrix([[{
            0: lambda: 2 * d,
            1: lambda: -1
        }.get(abs(i - j), lambda: 0)() for j in range(1, n)] for i in range(1, n)])
        return sm.bmat(res) if isinstance(res, list) and isinstance(res[0], list) else res
    return generate(d)

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.


    Attributes
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension
    """

    def __init__(self, d, n):
        """
        Parameters
        ----------
        d : int
            Dimension of the space
        n : int
            Number of intervals in each dimension

        Raises
        ------
        ValueError, if d < 1 or n < 2
        """
        if not d >= 1:
            raise ValueError("d must satisfy >= 1")
        if not n >= 2:
            raise ValueError("n must satisfy >= 2")

        self.d = d
        self.n = n
        self.extend = (n - 1) ** d
        self.data = construct(d, n)

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            block_matrix in a sparse data format
        """
        return self.data.tocsr()

    def eval_zeros(self):
        """ Returns the (absolute and relative) numbers of (non-)zero elements
        of the matrix. The relative number of the (non-)zero elements are with
        respect to the total number of elements of the matrix.

        Returns
        -------
        int
            number of non-zeros
        int
            number of zeros
        float
            relative number of non-zeros
        float
            relative number of zeros
        """
        sp = self.get_sparse()
        total_elems = self.extend ** 2
        non_zeros = sp.nnz
        zeros = total_elems - non_zeros
        rel_non_zeros = non_zeros / total_elems
        rel_zeros = zeros / total_elems

        return non_zeros, zeros, rel_non_zeros, rel_zeros

    # ----- ----- ----- -----

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form pr * A * pc = l * u

        Returns
        -------
        pr : scipy.sparse.csr_matrix
            row permutation matrix of LU-decomposition
        l : scipy.sparse.csr_matrix
            lower triangular unit diagonal matrix of LU-decomposition
        u : scipy.sparse.csr_matrix
            upper triangular matrix of LU-decomposition
        pc : scipy.sparse.csr_matrix
            column permutation matrix of LU-decomposition
        """
        lu = sla.splu(self.data)

        # convert the triangular matrix to csr (from csc)
        l, u = lu.L.tocsr(), lu.U.tocsr()

        # reconstruct and convert the permutation matrix to csr
        pr, pc = np.zeros((self.extend, self.extend)), np.zeros((self.extend, self.extend))
        pr[lu.perm_r, np.arange(self.extend)], pc[np.arange(self.extend), lu.perm_c] = 1, 1
        pr, pc = sm.csr_matrix(pr), sm.csr_matrix(pc)

        return pr, l, u, pc

    def eval_zeros_lu(self):
        """ Returns the absolute and relative numbers of (non-)zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        int
            Number of zeros
        float
            Relative number of non-zeros
        float
            Relative number of zeros
        """
        _, l, u, _ = self.get_lu()

        # absolute number of zeros for LU
        print(l.count_nonzero())
        print(u.count_nonzero())
        print()
        nnz = l.count_nonzero() + u.count_nonzero()
        nz = 2 * self.extend - nnz

        # absolute number of zeros for A
        mat_nnz = self.data.count_nonzero()
        mat_nz = self.extend - mat_nnz

        return nnz, nz, float(mat_nnz / nnz), float(mat_nz / nz)



    # should probably be moved out
    @staticmethod
    def test_lu(pr, l, u, pc):
        return pr.transpose() * l * u * pc.transpose()


def main():
    """ Internal demo-function for testing (or demoing) this module. """
    def demo_construction(d, n):
        """
        Demos block matrix for given dimension `d` and number of grid points `n` onto the terminal.

        Parameters:
        -----------
        d : int
            Dimension of the space
        n : int
            Number of intervals in each dimension
        """
        mat = BlockMatrix(d, n)
        print("# Demo BlockMatrix(d={}, n={}) in R^{{{}x{}}}".format(d, n, mat.extend, mat.extend))
        print("#==============================================")
        # print(mat.data)
        # print("#----------------------------------------------")
        sp = mat.get_sparse()
        print(sp.toarray())
        non_zeros, zeros, rel_non_zeros, rel_zeros = mat.eval_zeros()
        print("# non-zeros : {} (total number of non-zero values)".format(non_zeros))
        print("# zeros     : {} (total number of zero values)".format(zeros))
        print("% non-zeros : {:.2} (relative non-zero values)".format(rel_non_zeros))
        print("% zeros     : {:.2} (relative zero values)".format(rel_zeros))
        print()

    def demo_lu(d, n):
        mat = BlockMatrix(d, n)
        pr, l, u, pc = mat.get_lu()
        print("For the LU-Decomposition, Pr * A * Pl = LU, we have:\n")
        print("Row Permutation Matrix (Pr):\n{}".format(pr))
        print("The Matrix (A):\n".format(mat.data))
        print("Column Permutation Matrix (Pl):\n{}".format(pc))
        print("Lower Triangular Matrix (L):\n{}".format(l))
        print("Upper Triangular Matrix (U):\n{}".format(u))


    for d in [1, 2, 3]:
        for n in [2, 3, 4, 5]:
            # demo_construction(d, n)
            pass


    mat = BlockMatrix(2, 3)
    pr, l, u, pc = mat.get_lu()
    print(pr.toarray())
    print()
    print(mat.data.toarray())
    print()
    print(pc.toarray())
    print()
    print(l.toarray())
    print()
    print(u.toarray())
    print()
    # re = mat.test_lu(pr, l, u, pc)
    # print(re.toarray())
    nnz, nz, rel_nnz, rel_nz = mat.eval_zeros_lu()
    print("{}, {}, {}, {}".format(nnz, nz, rel_nnz, rel_nz))




if __name__ == "__main__":
    main()
