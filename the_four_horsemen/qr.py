#! /usr/bin/env python3
"""
This module implements QR-decomposition and linear regression.
"""

import numpy as np
from scipy import linalg

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

def qr(A):
    """
    This function computes the QR-decomposition.

    Arguments:
        A (ndarray): Two dimensional ndarray.

    Returns:
        (ndarray): Two dimensional ndarray of the QR-decomposition, Q.
        (ndarray): Two dimensional ndarray of the QR-decomposition, R.
    """
    # Rounding should not compromise the example given in the assignment as the data there have
    # less than six decimal places.
    return np.round(linalg.qr(A)[0], 12), np.round(linalg.qr(A)[1], 12)

def full_rank(A):
    """
    A boolean-function to check if the matrix has full column rank.

    Arguments:
        A (ndarray): Two dimensional ndarray.

    Returns:
        (boolean): True if A has full column rank and False otherwise.
    """
    R = qr(A)[1]
    for i in range(len(R[0])):
        if R[i][i] == 0:
            return False
    return True

def solve_QR(A, b):
    """
    Solves the equation Ax = b for x with the QR-decomposition.

    Arguments:
        A (ndarray): Two dimensional ndarray. Should have full column rank.
        b (ndarray): One dimensional ndarray.

    Returns:
        (ndarray): One dimensional ndarray for x.
    """
    Q, R = qr(A)
    z1 = np.matmul(np.transpose(Q), b)[0:A.shape[1]] # Q^T * [z1, z2] = b
    R1 = R[0:A.shape[1]]                             # R = [R1, 0]
    return linalg.solve_triangular(R1, z1)

def norm(A, b):
    """
    Computes the normed error induced by solving the linear equation with the function above.

    Arguments:
        A (ndarray): Two dimensional ndarray. Should have full column rank.
        b (ndarray): One dimensional ndarray.

    Returns:
        (float): The normed difference between Ax and b.
    """
    Ax = np.matmul(A, solve_QR(A, b))
    return linalg.norm(Ax - b)

def condition(A):
    """
    Calculates the condition of A and A^T * A.

    Arguments:
        A (ndarray): Two dimensional ndarray.

    Returns:
        (float, float): The condition of A and A^T * A.
    """
    sqr = np.matmul(np.transpose(A), A)
    return np.linalg.cond(A), np.linalg.cond(sqr)

def input_data(file_name, indices=None):
    """
    Takes a properly formated text file and builds a two dimensional ndarray. Additionally, a list
    of indices can be passed to return specific entries.

    Arguments:
        file_name (String): The path and the name of the text file.
        indices (list): List of indices.

    Returns:
        (ndarray): The entries of the matrix with the passed indices. If the argument indices were
                   left out, this returns the entire matrix.
    """
    with open(file_name, 'r') as f:
        data = np.array([[int(num) for num in line.split(',')] for line in f])
    return np.array([data[i][j] for (i, j) in indices]) if indices is not None else data

def draw(data):
    """
    Draws the plot of the linear regression according to the data passed.

    Arguments:
        data (ndarray): Two dimensional ndarray.
    """
    (a, b) = solve_QR(np.array([[1, i] for i in data[:, 1]]), np.array(data[:, 0]))
    p1 = data[:, 1]
    p0 = [a + b * x for x in p1]
    plt.plot(p0, p1)
    plt.scatter(data[:, 0], data[:, 1], color="r")

    plt.xlabel('$p_0$', fontsize=18)
    plt.ylabel('$p_1$', fontsize=18)
    plt.legend(fontsize=18)
    plt.title("Graph von $p_0$ und $p_1$", fontsize=24)
    plt.show()

def norm_residuals(data):
    """
    Computes the residuals for the solution with and without p2.

    Arguments:
        data (ndarray): Two dimensional ndarray.

    Returns:
        (float, float): The residuals for Ax = b without taking p2 into consideration and with.
    """
    b = data[:, 0] # take the first column as vector
    A_ab = np.array([[1, i] for i in data[:, 1]])
    A_abc = np.array([[1, i, j] for (i, j) in zip(data[:, 1], data[:, 2])])
    return norm(A_ab, b), norm(A_abc, b)

def main():
    """
    The main-function to demonstrate the capabilities of the module.
    """
    # pylint: disable=bad-whitespace
    A = np.transpose([[2,   3,  5, 7,  11],
                      [13, 17, 19, 23, 29],
                      [31, 37, 41, 43, 47]])

    b = [1, 2, 3, 4, 5]

    # We start by printing A and b.
    print("Matrix A =\n{}\n\nVector b =\n{}".format(A, b))
    print("\n===== ===== =====\n")
    # Proceed by printing the QR-decomposition of A.
    Q, R = qr(A)
    print("For the QR-decomposition, we have\nQ =\n{}\n\nR =\n{}".format(Q, R))
    print("\n===== ===== =====\n")
    # Has the matrix A full column self.data[:,1]rank?
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
    print("\n===== ===== =====\n")

    # The following is the demonstration for the given example.
    print("This module is also able to draw the plot for the given example.\n")
    data = input_data("assignment/pegel.txt")
    draw(data)
    print("The norm of the residuals are:")
    print("without p2: {}\nwith p2: {}".format(*norm_residuals(data)))
    print("The condition: {}".format(condition(data)))

    print("This module is also able to draw the plot for the given example.\n")
    data = input_data("assignment/pegel1.txt")
    draw(data)
    print("The norm of the residuals are:")
    print("without p2: {}\nwith p2: {}".format(*norm_residuals(data)))
    print("The condition: {}".format(condition(data)))

    print("This module is also able to draw the plot for the given example.\n")
    data = input_data("assignment/pegel2.txt")
    draw(data)
    print("The norm of the residuals are:")
    print("without p2: {}\nwith p2: {}".format(*norm_residuals(data)))
    print("The condition: {}".format(condition(data)))

if __name__ == "__main__":
    main()
