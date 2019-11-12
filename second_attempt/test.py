#! /usr/bin/env python3
import numpy as np
import functools 

def count_elements(block_matrix, a=0):
    """ Counts the number of coefficients in the matrix. """
    return functools.reduce(lambda a, x: a + (1 if type(x) != list else count_elements(x)), block_matrix, 0)

def depth(block_matrix):
    """ Computes the depth of a recursive block matrix. """
    return 1 if type(block_matrix) != list else 1 + depth(max(block_matrix, key = lambda p: depth(p)))

def dot_graph(matrix):
    """ Constructs the graph in dot-format. See: https://en.wikipedia.org/wiki/DOT_(graph_description_language) """
    def dot(node, depth):
        subs = functools.reduce(lambda a, n: a + ("" if type(n) != list else dot(n, depth + 1)), node, "")
        return functools.reduce(lambda a, n: a + "\t\"{}\" -> \"{}\";\n".format(node, n), node, "") + subs
    return "digraph {{\n{}\n}}\n".format(dot(matrix, 0))

def construct(d, n):
    """
    Constructs A(l, d) in R^{(n-1)^l x (n-1)^l}
    """
    def generate(l, global_i=1, global_j=1):
        def construct_A1():
            """ Constructs a A1 matrix in R^{(n-1)x(n-1)}. """
            def row(i):
                def entry(j):
                    if i == j:
                        return 2 * d
                    elif j == i - 1 or j == i + 1:
                        return -1
                    else:
                        return 0
                return [ entry(j) for j in range(1, n) ]
            return [ row(i) for i in range(1, n) ]
        def row(i):
            def entry(j):
                def null(k):
                    return [ [0 for j in range(1, k + 1)] for i in range(1, k + 1)]
                def minus_I(k):
                    return [ [(-1 if i == j else 0) for j in range(1, k + 1)] for i in range(1, k + 1)] # XXX
                if i == j:
                    return generate(l - 1)
                elif j == i - 1 or j == i + 1:
                    return minus_I((n - 1) ** (l-1))
                else:
                    return null((n - 1) ** (l-1))
            return [ entry(j) for j in range(1, n) ]
        if l > 1:
            return [ row(i) for i in range(1, n) ]
        elif l == 1:
            return construct_A1()
        else:
            raise ValueError("Invalid parameter l={0}.".format(l))
    if not (n >= 2):
        raise ValueError("n must be >= 2")
    return generate(d, 1, 1)

# hard_code_testing()
A_n = construct(2, 5)

