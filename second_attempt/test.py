#! /usr/bin/env python3
import functools 
import numpy as np
from scipy import sparse as sm

def count_elements(block_matrix, a=0):
    """ Counts the number of coefficients in the matrix. """
    return functools.reduce(lambda a, x: a + (1 if type(x) != list else count_elements(x)), block_matrix, 0)

def depth(block_matrix):
    """ Computes the depth of a recursive block matrix. """
    return 1 if type(block_matrix) != list else 1 + depth(max(block_matrix, key = lambda p: depth(p)))

def dot_graph(matrix):
    """ Constructs the graph in dot-format. See: https://en.wikipedia.org/wiki/DOT_(graph_description_language) """
    visited_leaves = []
    def adot(node, next_id, depth):
        s = ""
        if type(node) == list:
            my_id = next_id
            s += "\t{} [label=\"{}\"];\n".format(my_id, str(node))
            if depth < 200:
                for n in node:
                    if type(n) == list:
                        s += "\t{} -> {};\n".format(my_id, next_id + 1)
                        x, y = adot(n, next_id + 1, depth + 1)
                        s += x;
                        next_id = y
        # elif visited_leaves.count(node) == 0:
        #     visited_leaves.append(node)
        #     s += "\t{} [label=\"{}\"];\n".format(next_id, node)
        return s, next_id
    # def dot(node, depth):
    #     subs = functools.reduce(lambda a, n: a + ("" if type(n) != list else dot(n, depth + 1)), node, "")
    #     return functools.reduce(lambda a, n: a + "\t\"{}\" -> \"{}\";\n".format(node, n), node, "") + subs
    return "digraph {{\n{}\n}}\n".format(adot(matrix, 0, 0)[0])

def construct(d, n):
    """
    Constructs A(l, d) in R^{(n-1)^l x (n-1)^l}.
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
                    return -1 * sm.identity((n - 1) ** (l-1))
                else:
                    return null_matrix((n - 1) ** (l-1))
            return [ entry(j) for j in range(1, n) ]
        return [ row(i) for i in range(1, n) ] if l > 1 else construct_A1()
    if not (d >= 1):
        raise ValueError("d must satisfy >= 1")
    if not (n >= 2):
        raise ValueError("n must satisfy >= 2")
    return generate(d)

# hard_code_testing()
A_n = construct(2, 5)
print(sm.bmat(A_n).toarray())
print(A_n)

