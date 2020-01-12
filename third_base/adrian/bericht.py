"""
Dieses Programm liefert die Ausgaben des Berichtes zu Serie 3.2
des Projektpraktikums WS 19/20.

Datum: 02.01. 2020
Autoren: hollerju (589259), salihiad (572395)
"""
import numpy as np
import block_matrix
import linear_solvers
import rhs
import functions
import scipy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits import mplot3d

def u(x,y):
    """Demonstrationsfunktion zum Plotten der analytischen Lsg. des Poisson-Pro-
    blems.

    Input
    -----
    x : numpy.float64
    erste Koordinate
    y : numpy.float64
    zweite Koordinate
    """
    kappa = 10
    return x*np.sin(np.pi * kappa * x) * y * np.sin(np.pi * kappa * y)

def plot_analytical_3d(n, u):
    """Plottet die analytische Lösung u des Poisson Problems zur Funktion des
    Übungsblattes für d = 2.

    Input
    -----
    n : int
    Die Feinheit der Diskretisierung
    u : callable
    die analytische Lösung des Poisson-Problems
    """
    ax = plt.axes(projection="3d")
    x = np.linspace(1/n, (n-1)/n, n-1)
    y = np.linspace(1/n, (n-1)/n, n-1)
    X, Y = np.meshgrid(x, y)
    Z = u(X, Y)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Plot der analytischen Lösung des Poisson-Problems für \
    n = " + str(n))
    plt.show()

def plot_approximate_3d(n, f):
    """Plottet die approximierte Lösung des Poisson Problems zu einer gegebenen
    Funktion f mit Diskretisierung n.

    Input
    -----
    n : int
    Die Feinheit der Diskretisierung
    f : callable
    die Funktion f, die gleich dem negativen Laplace Operators der zu
    approximierenden Funktion ist
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = []
    y = []
    for i in range(n-1):
        for j in range(n-1):
            x.append(i/n)
            y.append(j/n)

    b = rhs.rhs(2, n, f)
    matrix = block_matrix.BlockMatrix(2, n)
    z = linear_solvers.solve_lu(*matrix.lu, b)

    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Plot der approximierten Lösung des Poisson-Problems für \
    n = " + str(n))
    plt.show()


def main():
    """Die main-Methode zur Demonstration der Module.

    Input
    -----
    None
    """
    d_1 = 1
    d_2 = 2
    d_3 = 3
    n_error = 15
    n_hilbert = 20
    n_solution = 4
    n_sparsity = 15
    test = np.array([1,2,3])
    values = range(3, n_error)
    matrix = block_matrix.BlockMatrix(d_1, n_error)
    list_1 = rhs.compute_error_list(d_1, n_error, functions.f_1, functions.u_1)
    list_2 = rhs.compute_error_list(d_2, n_error, functions.f_2, functions.u_2)
    list_3 = rhs.compute_error_list(d_3, n_error, functions.f_3, functions.u_3)
    rhs.plot_all_errors(values, list_1, list_2, list_3)
    block_matrix.graph_cond_hilbert(n_hilbert)
    plot_analytical_3d(n_solution, u)
    plot_approximate_3d(n_solution, functions.f_2)
    matrix.graph_abs_non_zero(n_sparsity)

if __name__ == '__main__':
    main()
