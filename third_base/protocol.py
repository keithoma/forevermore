""" This module demonstrates the experiments used in the protocol.
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
# from mpl_toolkits.mplot3d import Axes3D

import block_matrix
import linear_solvers
import rhs
# import functions

matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

def u(v, k=5):
    """ Example function with k = 5.0.

    Parameters:
    v : list
        The vector containing the grid points. Must not be np.array.
    k : float
        A constant.

    Returns:
    float
        The solution to the equation sum_{l = 1}^d x_l * sin(k * pi * x_l)
    """
    _ = 1
    vector_size = len(v) if isinstance(v, list) else 1
    for l in range(vector_size):
        _ = _ * v[l] * np.sin(k * np.pi * v[l])
    return _

def f(v, k=5):
    """ The derivative of above.

    Parameters:
    v : list
        The vector containing the grid points. Must not be np.array.
    k : float
        A constant.

    Returns:
    float
        The solution to the derivative.
    """
    if len(v) == 1:
        return k * np.pi * (k * np.pi * v[0] * np.sin(k * np. pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))

    elif len(v) == 2:
        x1, x2 = v[0], v[1]
        kpi = k * np.pi

        sum1 = kpi * x2 * np.sin(kpi * x2) * (kpi * x1 * np.sin(kpi * x1) - 2 * np.cos(kpi * x1))
        sum2 = kpi * x1 * np.sin(kpi * x1) * (kpi * x2 * np.sin(kpi * x2) - 2 * np.cos(kpi * x2))
        return sum1 + sum2

    elif len(v) == 3:
        x1, x2, x3 = v[0], v[1], v[2]
        kpi = k * np.pi

        s1 = kpi * x2 * x3 * np.sin(kpi * x2) * np.sin(kpi * x3) * (2 * np.cos(kpi * x1) - kpi * x1 * np.sin(kpi * x1))
        s2 = kpi * x1 * x3 * np.sin(kpi * x1) * np.sin(kpi * x3) * (2 * np.cos(kpi * x2) - kpi * x2 * np.sin(kpi * x2))
        s3 = kpi * x1 * x2 * np.sin(kpi * x1) * np.sin(kpi * x2) * (2 * np.cos(kpi * x3) - kpi * x3 * np.sin(kpi * x3))

        return -(s1 + s2 + s3)
    return 0.0

def plot_analytical_3d(N):
    """ This function draws the analytic solution for the poissons equation for d = 2.

    Parameters:
    N : int
        The number of grid points.
    """
    # create data
    grid = np.linspace(0.0, 1.0, N + 1, endpoint=True)
    # grid_length = len(grid)
    x_grid, y_grid = np.meshgrid(grid, grid)

    print("x_grid:\n{}\ny_grid:\n{}".format(x_grid, y_grid))

    z_axis = u([x_grid, y_grid])

    print("z axis (analytical)\n{}".format(z_axis))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if N >= 10:
        ax.plot_surface(x_grid, y_grid, z_axis, cmap=plt.cm.CMRmap, linewidth=0)
    else:
        ax.plot_surface(x_grid, y_grid, z_axis, linewidth=0)

    print("analytical")
    print(grid)
    print()

    lim = (-0.1, 1.1)
    ax.set_xlabel("$x$", fontsize=14), ax.set_xlim(lim)
    ax.set_ylabel("$y$", fontsize=14), ax.set_ylim(lim)
    ax.set_zlabel("$u(x, y)$", fontsize=14), ax.set_zlim(lim)
    plt.title("Plot of the Analytic Solution (N = " + str(N) + ")", fontsize=18)

    plt.show()

def plot_approximation_3d(N):
    """ This function draws the analytic solution for the poissons equation for d = 2.

    Parameters:
    N : int
        The number of grid points.
    """
    grid = np.linspace(0.0, 1.0, N + 1, endpoint=True)
    # grid_length = len(grid)
    x_grid, y_grid = np.meshgrid(grid, grid)

    b = linear_solvers.solve_lu(*block_matrix.BlockMatrix(2, N).get_lu(), rhs.rhs(2, N, f))
    z_grid = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            if i != 0 and i != N and j != 0 and j != N:
                z_grid[i][j] = b[0]
                b = np.delete(b, 0)

    print("z axis (approximation)\n{}".format(z_grid))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if N >= 10:
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=plt.cm.CMRmap, linewidth=0)
    else:
        ax.plot_surface(x_grid, y_grid, z_grid, linewidth=0)

    lim = (-0.1, 1.1)
    ax.set_xlabel("$x$", fontsize=14), ax.set_xlim(lim)
    ax.set_ylabel("$y$", fontsize=14), ax.set_ylim(lim)
    ax.set_zlabel("$\hat{u}(x, y)$", fontsize=14), ax.set_zlim(lim)
    plt.title("Plot of the Approximated Solution (N = " + str(N) + ")", fontsize=18)

    plt.show()

def main():
    """ The main function which reproduces the results of the protocol.
    """
    max_n = 15 # the following plots are drawn for upto this number

    rhs.draw_error(max_n)
    block_matrix.draw_cond(max_n)
    rhs.draw_hilbert_cond(max_n)
    block_matrix.draw_nonzero(max_n)

    # change the number in parentheses to change the number of grid points
    plot_analytical_3d(6)
    plot_approximation_3d(6)
    plot_analytical_3d(40)
    plot_approximation_3d(40)

if __name__ == '__main__':
    main()
