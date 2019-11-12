#! /usr/bin/env python3
import numpy as np
import functools 

def count_elements(block_matrix, a=0):
    """ Counts the number of coefficients in the matrix. """
    return functools.reduce(lambda a, x: a + (1 if type(x) != list else count_elements(x)), block_matrix, 0)

def depth(block_matrix):
    """ Computes the depth of a recursive block matrix. """
    return 1 if type(block_matrix) != list else 1 + depth(max(block_matrix, key = lambda p: depth(p)))

# def flatten(container):
#     for i in container:
#         if isinstance(i, (list,tuple)):
#             for j in flatten(i):
#                 yield j
#         else:
#             yield i

def construct(d, n):
    """
    Constructs A(l, d) in R^{(n-1)^l x (n-1)^l}
    """
    def generate(l, global_i=1, global_j=1):
        def construct_A1():
            """ Constructs a A1 matrix in R^{(n-1)*(n-1)}. """
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

def traverse_block_matrix(block_matrix):
    def traverse(block, i, j, depth):
        if type(block) == list and len(block) == 0:
            # -> empty (syntetic)
            return i, j
        elif type(block) == list and type(block[0]) == list:
            # -> list of rows of block matrices
            r, c = traverse(block[0], i, j, depth + 1)
            traverse(block[1:], r, j, depth + 1)
            return r, c
        if type(block) == list and type(block[0]) != list:
            # -> one row of elementary values
            traverse(block[0], i, j, depth + 1)
            traverse(block[1:], i, j + 1, depth + 1)
            return i + 1, j + len(block)
        else:
            # single elementary value
            print("[{}, {}] = {}".format(i, j, block))
            return i + 1, j + 1
    traverse(block_matrix, 0, 0, 0)

# =================
# TESTING SPACE ==
# ===============

def hard_code_testing():

    A_n = construct(2, 5)
    print(count_elements(A_n))

    print("=====")

    for item in A_n:
        for subitem in item:
            print(subitem)
            print("----")
    row_1 = []
    row_2 = []
    row_3 = []
    row_4 = []
    row_5 = []
    row_6 = []
    row_7 = []
    row_8 = []


    # i think the number of indices needed is (2d - 1)
    

    row_1.extend(A_n[0][0][0])
    row_1.extend(A_n[0][1][0])
    row_1.extend(A_n[0][2][0])
    row_1.extend(A_n[0][3][0])

    row_2.extend(A_n[0][0][1])
    row_2.extend(A_n[0][1][1])
    row_2.extend(A_n[0][2][1])
    row_2.extend(A_n[0][3][1])

    row_3.extend(A_n[0][0][2])
    row_3.extend(A_n[0][1][2])
    row_3.extend(A_n[0][2][2])
    row_3.extend(A_n[0][3][2])

    row_4.extend(A_n[0][0][3])
    row_4.extend(A_n[0][1][3])
    row_4.extend(A_n[0][2][3])
    row_4.extend(A_n[0][3][3])

    # -----

    row_5.extend(A_n[1][0][0])
    row_5.extend(A_n[1][1][0])
    row_5.extend(A_n[1][2][0])
    row_5.extend(A_n[1][3][0])

    row_6.extend(A_n[1][0][1])
    row_6.extend(A_n[1][1][1])
    row_6.extend(A_n[1][2][1])
    row_6.extend(A_n[1][3][1])

    row_7.extend(A_n[1][0][2])
    row_7.extend(A_n[1][1][2])
    row_7.extend(A_n[1][2][2])
    row_7.extend(A_n[1][3][2])

    row_8.extend(A_n[1][0][3])
    row_8.extend(A_n[1][1][3])
    row_8.extend(A_n[1][2][3])
    row_8.extend(A_n[1][3][3])

    print(row_1)
    print(row_2)
    print(row_3)
    print(row_4)
    print(row_5)
    print(row_6)
    print(row_7)
    print(row_8)

# hard_code_testing()
A_n = construct(2, 5)
print("Traversing:")
traverse_block_matrix(A_n)
