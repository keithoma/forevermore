#! /usr/bin/env python3
"""
This module implements the BlockMatrix class which is used to solve the Poisson problem.

Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

import numpy as np
from scipy import sparse as sm
from scipy.sparse import linalg as slina

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

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
        """ Auxilary function.

        Parameters
        ----------
        l : int
            The current index of the recursion.

        Returns
        -------
        sparse matrix
            The block matrix depending on l.
        """
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
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    extend : int
        The size of the matrix on one axis.
    data : sparse
        The matrix saved as SciPy sparse format. The format is appropriately chosen by SciPy.
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
            raise ValueError("d must satisfy >= 1.")
        if not n >= 2:
            raise ValueError("n must satisfy >= 2.")

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
        lu = slina.splu(sm.csc_matrix(self.data))

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
        nnz = l.count_nonzero() + u.count_nonzero() - self.extend
        nz = self.extend ** 2 - nnz

        # absolute number of zeros for A
        mat_nnz = self.data.count_nonzero()
        mat_nz = self.extend ** 2 - mat_nnz

        return nnz, nz, float(mat_nnz / nnz) if nnz != 0 else 1, float(mat_nz / nz) if nz != 0 else 1

    def get_cond(self):
        """ Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to max-norm
        """
        # if n is 2, then the matrix has the dimension 1x1
        if self.n == 2:
            return 1

        # SciPy likes csc more to calculate the inverse
        csc = sm.csc_matrix(self.data)
        return sm.linalg.norm(csc, np.inf) * sm.linalg.norm(sm.linalg.inv(csc), np.inf)

def draw_cond(max_n=15):
    """
    Draws the graph for the condition of A.

    Parameters
    ----------
    max_n : int
        The maximal n for which the plot should be drawn.
    """
    # get the values for the plot
    for d in [1, 2, 3]:
        plt.plot([x for x in range(2, max_n)],
                 [BlockMatrix(d, x).get_cond() for x in range(2, max_n)],
                 label="cond" + "$(A_" + str(d) + ")$")

    # finilize
    plt.xlabel('$N$', fontsize=14)
    plt.ylabel('$cond(A)$', fontsize=14)
    plt.legend(fontsize=18)
    plt.title("Plot of cond(A) depending on the dimension and size")
    plt.show()

def draw_nonzero(max_n=15):
    """
    Draws the graph for the nonzero entries of A.

    Parameters
    ----------
    max_n : int
        The maximal n for which the plot should be drawn.
    """
    # get the values for the plot
    for d in [1, 2, 3]:
        plt.loglog([x for x in range(2, max_n)],
                   [BlockMatrix(d, x).eval_zeros()[0] for x in range(2, max_n)],
                   label="nonzero elements of " + "$A_" + str(d) + "$",
                   linewidth=2)
        plt.loglog([x for x in range(2, max_n)],
                   [BlockMatrix(d, x).eval_zeros_lu()[0] for x in range(2, max_n)],
                   label="nonzero elements of " + "$LU_" + str(d) + "$",
                   linestyle='--', linewidth=2)

    # finilize
    plt.xlabel('n', fontsize=14)
    plt.ylabel('number of nonzeros', fontsize=14)
    plt.legend(fontsize=18)
    plt.title("Plot of nonzero elements of A and LU depending on the dimension and size")
    plt.show()

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
        print("#======================================================")
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
        """ Demonstration of the LU-decomposition.

        Parameters
        ----------
        d : int
            The dimension.
        n : int
            The number of grid points.
        """
        mat = BlockMatrix(d, n)
        pr, l, u, pc = mat.get_lu()
        print("Let d = {} and n = {}. For the LU-Decomposition, Pr * A * Pl = LU, we have:\n".format(d, n))
        print("Row Permutation Matrix (Pr):\n{}\n".format(pr.toarray()))
        print("The Matrix (A):\n{}\n".format(mat.data.toarray()))
        print("Column Permutation Matrix (Pl):\n{}\n".format(pc.toarray()))
        print("Lower Triangular Matrix (L):\n{}\n".format(l.toarray()))
        print("Upper Triangular Matrix (U):\n{}\n".format(u.toarray()))

    def demo_nnz(d, n):
        """ Demonstration of the nonzero counting method..

        Parameters
        ----------
        d : int
            The dimension.
        n : int
            The number of grid points.
        """
        mat = BlockMatrix(d, n)
        print("# NUMBER OF ZEROS LU DECOMPOSITION(d={}, n={}) in R^{{{}x{}}}".format(d, n, mat.extend, mat.extend))
        print("#======================================================")
        # demo_lu(d, n)
        non_zeros, zeros, rel_non_zeros, rel_zeros = mat.eval_zeros_lu()
        print("# non-zeros : {} (total number of non-zero values)".format(non_zeros))
        print("# zeros     : {} (total number of zero values)".format(zeros))
        print("% non-zeros : {:.2} (relative non-zero values)".format(float(rel_non_zeros)))
        print("% zeros     : {:.2} (relative zero values)".format(float(rel_zeros)))
        print()
        print("Cond(A) = {}".format(mat.get_cond()))
        print()

    for d in [1, 2]:
        for n in [2, 3, 4]:
            #demo_construction(d, n)
            #demo_nnz(d, n)
            pass

    #demo_lu(1, 5)
    #demo_lu(2, 4)

    print("We can also draw plots.")
    draw_cond()
    draw_nonzero()

if __name__ == "__main__":
    main()
