"""
Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

import functools as fp
import numpy as np
from scipy import sparse as sm

def count_elements(block_matrix, a=0):
    """ Counts the number of coefficients in the matrix. """
    return fp.reduce(lambda a, x: a + (1 if type(x) != list else count_elements(x)), block_matrix, 0)

def depth(block_matrix):
    """ Computes the depth of a recursive block matrix. """
    return 1 if type(block_matrix) != list else 1 + depth(max(block_matrix, key = lambda p: depth(p)))

def dot_graph(matrix):
    """ Constructs the graph in dot-format. See: https://en.wikipedia.org/wiki/DOT_(graph_description_language) """
    def dot(node, depth):
        subs = fp.reduce(lambda a, n: a + ("" if type(n) != list else dot(n, depth + 1)), node, "")
        return fp.reduce(lambda a, n: a + "\t\"{}\" -> \"{}\";\n".format(node, n), node, "") + subs
    return "digraph {{\n{}\n}}\n".format(dot(matrix, 0))

def construct(d, n):
    """
    Constructs block matrices arising from finite difference approximations
    of the Laplace operator.

    Attributes
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension

    Returns
    -------
    list
        A row-major list of block matrix entries suitable for feeding into SciPy's API. 
    """
    def generate(l):
        """ Helper function for constructing a (sub-) block matrix for given `l` parameter. """
        def construct_A1():
            """ Constructs a A1 matrix in R^{(n-1)x(n-1)}. """
            def row(i):
                """ Constructs the row at A_{i} """
                def entry(j):
                    """ Constructs an entry at A_{i,j} """
                    return 2 * d if i == j else -1 if j == i - 1 or j == i + 1 else 0
                return [ entry(j) for j in range(1, n) ]
            return sm.coo_matrix([ row(i) for i in range(1, n) ])
        def row(i):
            """ Constructs the row at A_{i} """
            def entry(j):
                """ Constructs an entry at A_{i,j} """
                def null_matrix(k):
                    """ Constructs a null-matrix of `k` rows and `k` columns.
                        Even though parameter `k` is not used, it is actually meant to be passed,
                        because before using SciPy, we have been doing everything our own,
                        and I want to preserve that right to move away from SciPy, also,
                        it's nice documentation to know what dimension this matrix is meant to be.
                    """
                    return None
                if i == j:
                    return generate(l - 1)
                elif j == i - 1 or j == i + 1:
                    return -1 * sm.identity((n - 1) ** (l - 1))
                else:
                    return null_matrix((n - 1) ** (l - 1))
            return [ entry(j) for j in range(1, n) ]
        res = [ row(i) for i in range(1, n) ] if l > 1 else construct_A1()
        return sm.bmat(res) if type(res) == list and type(res[0]) == list else res
    if not (d >= 1):
        raise ValueError("d must satisfy >= 1")
    if not (n >= 2):
        raise ValueError("n must satisfy >= 2")
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
        """
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
        return self.data

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

def test_main():
    """ internal demo-function for testing this. """
    def demo(d, n):
        """ Demos block matrix for given `d` and `n` onto the terminal. """
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
    for d in [1, 2, 3, 4]:
        for n in [2, 3, 4, 5]:
            demo(d, n)

if __name__ == "__main__":
    test_main()