#!/usr/bin/python3
"""
This simple module draws the plot of the number of (nonzero) entries of the matrix A_d. Seee the
accompanying protocol for more information.

Spewed into existence by:
Christian Parpart
Kei Thoma
"""

import matplotlib.pyplot as plt

def count_entries_dense(d, n):
    """
    Counts all entries of A_d.

    Params
    ------
    d : int
        The dimension of the Laplace equation.
    n : int
        The number of to be evaluated grid points of the domain.

    Returns
    -------
    int
        The number of all entries of A_d.
    """
    return (n - 1) ** (2 * d)

def count_entries_sparse(d, n):
    """
    Counts all nonzero entries of A_d.

    Params
    ------
    d : int
        The dimension of the Laplace equation.
    n : int
        The number of to be evaluated grid points of the domain.

    Returns
    -------
    int or None
        The number of all nonzero entries of A_d. If d is not 1, 2 or 3, it will return None.
    """
    if d == 1:
        return 3 * n - 5
    elif d == 2:
        return 5 * (n - 1) ** 2 - 4 * (n - 1)
    elif d == 3:
        return 7 * (n - 1) ** 3 - 6 * (n - 1) ** 2
    return None

def draw_amount_of_entries():
    """
    Draws the plot on double log scale for the number of counted entries of dense and sparse.
    """
    # values for x-axis
    x_axis = [i for i in range(2, 10002)]

    # values for y-axis
    dense = [[count_entries_dense(d, i) for i in x_axis] for d in (1, 2, 3)]
    sparse = [[count_entries_sparse(d, i) for i in x_axis] for d in (1, 2, 3)]

    # draw the plot
    for d in [0, 1, 2]:
        print(dense[d])
        print(sparse[d])
        plt.loglog(x_axis, dense[d], label="$f_{dense}^{(d=" + str(d + 1) + ")} (x)$", linestyle=":")
        plt.loglog(x_axis, sparse[d], label="$f_{sparse}^{(d=" + str(d + 1) + ")} (x)$")

    # graphics
    plt.title("Plot of Number of (Nonzero) Entries of $A_d$")
    plt.xlabel('n')
    plt.ylabel('Number of (Nonzero) Entries')
    plt.legend()
    plt.show()

def main():
    """
    Wow! This is the main-function which just calls the draw function.
    """
    draw_amount_of_entries()

if __name__ == "__main__":
    main()
