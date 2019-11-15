#! /usr/bin/env python3
"""
Author: Christian Parpart & Kei Thoma
Date: 2019-11-13
License: GPL-3
"""

import functools as fp
from scipy import sparse as sm

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
    """
        construct(d, n)
            generate(l)
                construct_A1()
                    row(i)
                        entry(j)
                row(i)
                    entry(j)
    """
    def generate(l):
        def construct_A1(): # pylint: disable=invalid-name
            def row(i):
                return [{
                    0: lambda: 2 * d,
                    1: lambda: -1
                }.get(abs(i - j), lambda: 0)() for j in range(1, n)]
            return sm.coo_matrix([row(i) for i in range(1, n)])
        def row(i):
            return [{
                0: lambda: generate(l - 1),
                1: lambda: -1 * sm.identity((n - 1) ** (l - 1))
            }.get(abs(i - j), lambda: None)() for j in range(1, n)]
        res = [row(i) for i in range(1, n)] if l > 1 else construct_A1()
        return sm.bmat(res) if isinstance(res, list) and isinstance(res[0], list) else res
    if not d >= 1:
        raise ValueError("d must satisfy >= 1")
    if not n >= 2:
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
