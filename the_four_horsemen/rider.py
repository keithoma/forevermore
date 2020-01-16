#! /usr/bin/env python3

import numpy as np
from scipy import linalg

def qr(A):
    # Rounding should not compromise the example given in the assignment as the data there have less than
    # six decimal places.
    return np.round(linalg.qr(A)[0], 6), np.round(linalg.qr(A)[1], 6)

def full_rank(A):
    R = qr(A)[1]
    for i in range(len(R[0])):
        if R[i][i] == 0:
            return False
    return True

def solve_QR(A, b):
    Q, R = qr(A)
    z1 = np.matmul(np.transpose(Q), b)[0:A.shape[1]] # Q^T * [z1, z2] = b
    R1 = R[0:A.shape[1]]                             # R = [R1, 0]
    return linalg.solve_triangular(R1, z1)

def norm(A, b):
    Ax = np.matmul(A, solve_QR(A, b))
    return linalg.norm(Ax - b)

def condition(A):
    sqr = np.matmul(np.transpose(A), A)
    return np.linalg.cond(A), np.linalg.cond(sqr)

def input_data(file_name, indices=None):
    with open(file_name, 'r') as f:
        data = [[int(num) for num in line.split(',')] for line in f]
    return [data[i][j] for (i, j) in indices] if indices is not None else data

def main():
    def demo(A, b):
        # We start by printing A and b.
        print("Matrix A =\n{}\n\nVector b =\n{}".format(A, b))
        print("\n===== ===== =====\n")
        # Proceed by printing the QR-decomposition of A.
        Q, R = qr(A)
        print("For the QR-decomposition, we have\nQ =\n{}\n\nR =\n{}".format(Q, R))
        print("\n===== ===== =====\n")
        # Has the matrix A full column rank?
        print("Has the matrix A full column rank?\n=> {}".format(full_rank(A)))
        print("\n===== ===== =====\n")
        # Solve Ax = b.
        print("We solve the equation Ax = b with the QR-decomposition.\nb =\n{}".format(solve_QR(A, b)))
        print("\n===== ===== =====\n")
        # Norm.
        print("The norm of (Ax - b) is\n{}".format(norm(A, b)))
        print("\n===== ===== =====\n")
        # Condition.
        cond_A, cond_sqr = condition(A)
        print("The condition of the matrix A is\n{}\n".format(cond_A))
        print("The condition of the matrix A^T * A is\n{}\n".format(cond_sqr))

    A = np.transpose([[2,   3,  5, 7,  11],
                      [13, 17, 19, 23, 29],
                      [31, 37, 41, 43, 47]])

    b = [1, 2, 3, 4, 5]
    demo(A, b)

    print(input_data("assignment/pegel.txt", [[0, 0], [0, 1]]))

if __name__ == "__main__":
    main()
