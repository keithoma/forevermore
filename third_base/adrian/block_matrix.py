"""
Program to create block-band-matrices for partial differential equations in
d dimensions (d = 1, 2, 3)

Author:  hollerju (589259), salihiad (572395)
Date: 26.11.2019

"""
import scipy.sparse
from scipy.sparse import linalg as sla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class BlockMatrix:
    """
    Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension

    Attributes
    ----------
    d : int
        dimension of the space
    n : int
        number of intervals in each dimension
    number_of_elements : int
                        number of elements in the matrix
    matrix : scipy.sparse.csr_matrix
            the matrix we want to construct as a csr_matrix
    """
    def __init__(self, d, n):
        """
        Initialises our class-object using the given parameters.

        Raises value error, if d is not in {1, 2, 3} or if n is not bigger than
        2
        """
        # dimension d has to be 1, 2 or 3
        if not d in [1, 2, 3]:
            raise ValueError
        # n has to be bigger than 2
        if n < 2:
            raise ValueError

        self.d = d
        self.n = n
        self.number_of_elements = (n - 1)**(2*d)
        self.matrix = self.get_sparse(d, d)
        self.lu = self.get_lu()

    def get_sparse(self, param_l=None, param_k=None):
        """
        Returns the block matrix for our parameters
        as a scipy.sparse.csr_matrix.

        Parameters:
        --------
        param_l : int
                the parameter l to create the block_matrix
        param_k : int
                the parameter k to create the diagonal

        Returns:
        -------
        scipy.sparse.csr_matrix
            block_matrix for the poisson problem

        Raises:
        -------
        ValueError, if param_l is smaller than 1
        """
        if param_l < 1:
            raise ValueError

        if param_l == 1:
            # create matrix with the size (n-1)x(n-1)
            # 2 on main-diagonal, -1 on the  "1. and (-1)." minor diagonals
            return scipy.sparse.diags([2 * param_k, -1, -1], [0, 1, -1],
                                      shape=(self.n-1, self.n-1), format="csr")

        # if l > 1, create the matrix recursively as a block-band-matrix.
        minus_id = self.negative_identity_matrix(param_l-1)
        # call recursively to calculate A_[l-1](k)
        matrix_a = self.get_sparse(param_l-1, param_k)

        # create block-diagonal-matrix
        # blocks = [[None] * (self.n - 1)] * (self.n-1)
        blocks = [[None for i in range(self.n-1)] for j in range(self.n-1)]

        # main-diagonal: blocks of matrix A
        for i in range(self.n-1):
            blocks[i][i] = matrix_a

        # minor diagonal filled with blocks of matrix -I_(l-1)
        for i in range(self.n-2):
            blocks[i+1][i] = minus_id
            blocks[i][i+1] = minus_id

        # return matrix as a scipy.sparse matrix
        return scipy.sparse.bmat(blocks, format="csr")


    def negative_identity_matrix(self, dimension):
        """
        Creates the matrix (-1)*identity_matrix for a given dimension

        Parameters:
        -----------
        dimension : int
                    the dimension d in which the matrix should be created (dxd)

        Returns:
        --------
            scipy.sparse.csr_matrix
                    the (-1)*identity_matrix in our dimension

        """
        return -scipy.sparse.identity((self.n - 1)**dimension, format="csr")


    def eval_zeros(self):
        """
        Returns the (absolute and relative) numbers of (non-)zero elements
        of the matrix. The relative number of the (non-)zero elements are with
        respect to the total number of elements of the matrix.

        Parameters:
        -----------
        None

        Returns
        -------
        non_zeros : int
        number of non-zeros
        zeros : int
        number of zeros
        rel_non_zeros : float
        relative number of non-zeros
        rel_zeros : float
        relative number of zeros
        """
        non_zeros = self.matrix.count_nonzero()
        zeros = self.number_of_elements - self.matrix.count_nonzero()
        rel_non_zeros = non_zeros / self.number_of_elements
        rel_zeros = zeros / self.number_of_elements
        return non_zeros, zeros, rel_non_zeros, rel_zeros

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
        # create csc_matrix for splu
        make_csc = scipy.sparse.csc_matrix(self.matrix)
        make_lu = sla.splu(make_csc)
        # Permutationsmatrizen erstellen
        pr = scipy.sparse.csc_matrix(((self.n -1)**self.d, (self.n -1)**self.d))
        pc = scipy.sparse.csc_matrix(((self.n -1)**self.d, (self.n -1)**self.d))
        # DirtySolution but works
        tmp_1 = np.array(make_lu.perm_r)
        tmp_2 = np.array(make_lu.perm_c)
        pr[tmp_1, np.arange((self.n -1)**self.d)] = 1
        pc[np.arange((self.n-1)**self.d), tmp_2] = 1
        # Change to csr_matrix
        pr = scipy.sparse.csr_matrix(pr)
        pc = scipy.sparse.csr_matrix(pc)
        l = scipy.sparse.csr_matrix(make_lu.L)
        u = scipy.sparse.csr_matrix(make_lu.U)
        return pr, l, u, pc

    def eval_zeros_lu(self):
        """ Returns the absolute and relative numbers of (non-)zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.
        Returns
        -------
        abs_non_zero_LU : int
        Number of non-zeros
        abs_zero_LU : int
        Number of zeros
        rel_non_zero_LU : float
        Relative number of non-zeros
        rel_zero_LU : float
        Relative number of zeros
        """
        # calculate the total number of elements in LU
        total_entries_in_LU = 2*self.number_of_elements
        # get absolute number of nonzero elements in LU
        abs_non_zero_L = self.lu[1].count_nonzero()
        abs_non_zero_U = self.lu[2].count_nonzero()
        abs_non_zero_LU = abs_non_zero_L + abs_non_zero_U
        # find absolute number of elements in LU that are zero
        abs_zero_LU = total_entries_in_LU - abs_non_zero_LU
        # calculate relative number of (non) zero elements in LU
        rel_non_zero_LU = abs_non_zero_LU / total_entries_in_LU
        rel_zero_LU = abs_zero_LU / total_entries_in_LU
        return abs_non_zero_LU, abs_zero_LU, rel_non_zero_LU, rel_zero_LU

    def get_cond(self):
        """ Computes the condition number of the represented matrix.
        Returns
        -------
        float
        condition number with respect to the row sum norm
        """
        make_csc = scipy.sparse.csc_matrix(self.matrix)
        inverse = scipy.sparse.linalg.inv(make_csc)
        norm_matrix = scipy.sparse.linalg.norm(make_csc, np.inf)
        norm_inverse = scipy.sparse.linalg.norm(inverse, np.inf)
        return norm_matrix * norm_inverse

    def graph_cond(self, high_N):
        """
        Plots the condition of A^(d) in dependency of N.

        Parameter
        ---------
        high_N : int
        Highest N for which we want to plot the condition of A^(d)

        Return
        ------
        None, we plot the graph and show it
        """
        list_conditions = []
        real_dimension = self.d
        real_feinheit = self.n
        real_matrix = self.matrix
        plt.style.use("seaborn-white")
        plt.figure(figsize=(14, 7))
        plt.gcf().canvas.set_window_title("Graph der Kondition \
        von A^(d) in Abhängigkeit von N")
        plt.title("Kondition von A^(d)")
        axis = plt.gca()
        for d in range(1, 4):
            self.d = d
            for n in range(3, high_N):
                self.n = n
                self.matrix = self.get_sparse(self.d, self.d)
                list_conditions.append(self.get_cond())
            axis.plot(range(3, high_N), list_conditions, label="Kondition für \
d = " + str(d))
            list_conditions = []
        plt.xlabel("N-Werte")
        plt.ylabel("Kondition von A(d)")
        plt.grid(True)
        plt.legend()
        plt.show()
        self.n = real_feinheit
        self.d = real_dimension
        self.matrix = real_matrix

    def graph_abs_non_zero(self, high_N):
        """Plots the number of non-zero entries in the matrix A^(d) and LU in
        dependency of N

        Parameter
        ---------
        high_N : int
        Highest N for which we want to plot the number of nonzeros

        Return
        ------
        None, we plot the graph and show it
        """
        list_abs_non_zero = []
        list_non_zero_A = []
        real_dimension = self.d
        real_feinheit = self.n
        real_matrix = self.matrix
        real_number_of_elements = self.number_of_elements
        plt.style.use("seaborn-white")
        plt.figure(figsize=(14, 7))
        plt.gcf().canvas.set_window_title("Graph der Nicht-Null-Einträge der\
 Matrix A^(d) und ihrer LU Zerlegung")
        plt.title("Graph der Nicht-Null-Einträge von A^(d) und ihrer LU \
Zerlegung")
        axis = plt.gca()
        for d in range(1, 4):
            self.d = d
            for n in range(3, high_N):
                self.n = n
                self.matrix = self.get_sparse(self.d, self.d)
                self.lu = self.get_lu()
                self.number_of_elements = (self.n)**(2*self.d)
                list_abs_non_zero.append(self.eval_zeros_lu()[0])
                list_non_zero_A.append(self.eval_zeros()[0])
            axis.plot(range(3, high_N), list_non_zero_A, label="Nicht-Null-\
Einträge für A bei d = " + str(d), linestyle="dashed")
            axis.plot(range(3, high_N), list_abs_non_zero, label="Nicht-Null-\
Einträge von LU für d = " + str(d))
            list_abs_non_zero = []
            list_non_zero_A = []
        plt.xlabel("n-Werte")
        plt.ylabel("Nicht-Null-Einträge")
        axis.set_xscale('log')
        axis.set_yscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()
        self.n = real_feinheit
        self.d = real_dimension
        self.matrix = real_matrix
        self.number_of_elements = real_number_of_elements

def graph_cond_hilbert(high_n):
    condition = []
    for i in range(3, high_n):
        hilbert = scipy.linalg.hilbert(i)
        condition.append(np.linalg.cond(hilbert, np.inf))
    plt.style.use("seaborn-white")
    plt.figure(figsize=(14, 7))
    plt.gcf().canvas.set_window_title("Graph der Kondition \
    von A^(d) in Abhängigkeit von N")
    plt.title("Kondition von A^(d)")
    axis = plt.gca()
    axis.plot(range(3, high_n), condition, label="Kondition der Hilbertmatrix\
in Abhängigkeit von n")
    plt.xlabel("n-Werte")
    plt.ylabel("Kondition der Hilbertmatrix")
    axis.set_yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
