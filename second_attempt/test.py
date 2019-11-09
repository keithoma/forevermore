#! /usr/bin/env python3
import numpy as np

def null(n):
    return [ [0 for j in range(1, n + 1)] for i in range(1, n + 1)]

def identity(n):
    return [ [(1 if i == j else 0) for j in range(1, n + 1)] for i in range(1, n + 1)]

def minus_I(n):
    return [ [(-1 if i == j else 0) for j in range(1, n + 1)] for i in range(1, n + 1)]

def construct_A1(d, n):
    """ Constructs a A1 matrix in R^{(n-1)*(n-1)}. """
    def row(j, d):
        def entry(i):
            if i == j:
                return 2 * d
            elif j == i + 1 or j == i - 1:
                return -1
            else:
                return 0
        return [ entry(j) for j in range(1, n) ]
    return [ row(i, d) for i in range(1, n) ]

def count_elements(multi_list, a=0):
    for item in multi_list:
        a = a + 1 if type(item) != list else count_elements(item, a)
    return a

# def flatten(container):
#     for i in container:
#         if isinstance(i, (list,tuple)):
#             for j in flatten(i):
#                 yield j
#         else:
#             yield i

def construct(d, l, n):
    """
    Constructs A(l, d) in R^{(n-1)^l x (n-1)^l}
    """
    if not (n >= 2):
        raise ValueError("n must be >= 2")
    def row(i):
        def entry(j):
            if i == j:
                return construct(d, l - 1, n)
            elif j == i - 1 or j == i + 1:
                return minus_I((n - 1) ** (l-1))
            else:
                return null((n - 1) ** (l-1))
        return [ entry(j) for j in range(1, n) ]
    if l > 1:
        return [ row(i) for i in range(1, n) ]
    elif l == 1:
        return construct_A1(d, n)
    else:
        raise ValueError("Invalid parameter l={0}.".format(l))

A_n = construct(2, 2, 3)
print(A_n)
#print(reshape(A_n, 4))
print(count_elements(A_n))
