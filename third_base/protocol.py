import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

import block_matrix
import linear_solvers
import rhs
import functions

def u(v, k=5.0):
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

def f(v, k=5.0):
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
        sum1 = k * np.pi * v[1] * np.sin(k * np.pi * v[1]) * (k * np.pi * v[0] * np.sin(k * np.pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))
        sum2 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * (k * np.pi * v[1] * np.sin(k * np.pi * v[1]) - 2 * np.cos(k * np.pi * v[1]))
        return sum1 + sum2

    elif len(v) == 3:
        sum1 = k * np.pi * v[1] * np.sin(k * np.pi * v[1]) * k * np.pi * v[2] * np.sin(k * np.pi * v[2])
        sum1 = sum1 * (k * np.pi * v[0] * np.sin(k * np.pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))

        sum2 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * k * np.pi * v[2] * np.sin(k * np.pi * v[2])
        sum2 = sum1 * (k * np.pi * v[1] * np.sin(k * np.pi * v[1]) - 2 * np.cos(k * np.pi * v[1]))

        sum3 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * k * np.pi * v[1] * np.sin(k * np.pi * v[1])
        sum3 = sum1 * (k * np.pi * v[2] * np.sin(k * np.pi * v[2]) - 2 * np.cos(k * np.pi * v[2]))
        return sum1 + sum2 + sum3

def plot_analytical_3d(N):
    """ This function draws the analytic solution for the poissons equation for d = 2.

    Parameters:
    N : int
        The number of grid points.
    """
    # create data
    grid = np.linspace(0.0, 1.0, N, endpoint=False)[1:]
    grid_length = len(grid)
    x_grid, y_grid = np.meshgrid(grid, grid)
    z_axis = u([x_grid, y_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if N >= 10:
        ax.plot_surface(x_grid, y_grid, z_axis, cmap=plt.cm.CMRmap, linewidth=0)
    else:
        ax.plot_surface(x_grid, y_grid, z_axis, linewidth=0)

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_zlabel("$u(x, y)$", fontsize=14)
    plt.title("Plot of the Analytic Solution (N = " + str(N) + ")", fontsize=18)

    plt.show()

def plot_approximation_3d(N):
    """ This function draws the analytic solution for the poissons equation for d = 2.

    Parameters:
    N : int
        The number of grid points.
    """
    x = [i / N for i in range(N - 1) for j in range(N - 1)]
    y = [j / N for i in range(N - 1) for j in range(N - 1)]
    z_axis = linear_solvers.solve_lu(*block_matrix.BlockMatrix(2, N).get_lu(), rhs.rhs(2, N, f))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if N >= 10:
        ax.plot_trisurf(x, y, z_axis, linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
    else:
        ax.plot_trisurf(x, y, z_axis, linewidth=0.2, antialiased=True)

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_zlabel("$\hat{u}(x, y)$", fontsize=14)
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
    plot_analytical_3d(4)
    plot_approximation_3d(4)
    plot_analytical_3d(30)
    plot_approximation_3d(30)

if __name__ == '__main__':
    main()
