import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

import block_matrix
import functions
import linear_solvers
import rhs

matplotlib.use('TkAgg')

class ComputationSettings:
    def __init__(self, d=1, n=2):
        self.d = d
        self.n = n
        self.f = functions.f
        self.u = functions.u
        self.x0 = np.array([0 for _ in range((n - 1) ** d)])
        self.params = dict(eps=1e-8, max_iter=1000, min_red=1e-4)
        self.omega = 1.5

    def change(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if "x0" not in kwargs:
            self.x0 = np.array([0 for _ in range((self.n - 1) ** self.d)])
        return self

    # not the most elegant way, but w/e
    def change_eps(self, k):
        if k is not None:
            self.params["eps"] = 1 / (self.n + 1) ** k
        return self

class DrawSettings:
    def __init__(self):
        self.dlist = [1, 2, 3]
        self.max_n = 15
        self.f = functions.f
        self.u = functions.u
    
    def change(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

# aliasing to type less
cs, ds = ComputationSettings, DrawSettings

# auxilary functions
def solve_poisson_sor(cs):
    A = block_matrix.BlockMatrix(cs.d, cs.n).data.tocsr()
    b = rhs.rhs(cs.d, cs.n, cs.f)
    print("d = {}, n = {}, omega = {}".format(cs.d, cs.n, cs.omega))
    return linear_solvers.solve_sor(A, b, cs.x0, params=cs.params, omega=cs.omega)

def true_solution(cs):
    grid = np.linspace(0.0, 1.0, cs.n, endpoint=False)[1:]
    if cs.d == 1:
        true_solution = np.array([cs.u([x]) for x in grid])
    elif cs.d == 2:
        true_solution = np.array([cs.u([y, x]) for x in grid for y in grid])
    elif cs.d == 3:
        true_solution = np.array([cs.u([z, y, x]) for x in grid for y in grid for z in grid])
    else:
        raise ValueError("d must be 1, 2 or 3.")
    return true_solution

# plotting functions
def plot_omega(cs):
    # values
    omega_list = np.linspace(1.0, 1.9, 10)
    number_iterations = [len(solve_poisson_sor(cs.change(omega=omega))[1]) for omega in omega_list]

    # plot
    plt.plot(omega_list, number_iterations)
    plt.xlabel("$\omega$", fontsize=28)
    plt.ylabel("Number of Iterations", fontsize=28)
    plt.title("Number of Iterations for $d = {}$ and $n = {}$".format(cs.d, cs.n), fontsize=24)
    plt.show()

def plot_absolute_error(cs):
    # values
    solution_list = solve_poisson_sor(cs)[1]
    exact = true_solution(cs)

    x_axis = [n for n in range(len(solution_list))]
    y_axis = [max([abs(ai - bi) for ai, bi in zip(exact, iteration)]) for iteration in solution_list]

    # draw plot
    plt.plot(x_axis, y_axis)
    plt.show()

def compare_lu_sor(cs, ds, k=None):
    def compute_error_sor(d, n, k):
        iteration = solve_poisson_sor(cs.change(d=d, n=n).change_eps(k))[1][-1]
        exact = true_solution(cs.change(d=d, n=n))
        return max([abs(ai - bi) for ai, bi in zip(exact, iteration)])

    def compute_error_lu(d, n):
        algorithm = linear_solvers.solve_lu(*block_matrix.BlockMatrix(d, n).get_lu(),
                                            rhs.rhs(d, n, functions.f))
        exact = true_solution(cs.change(d=d, n=n))
        return max([abs(ai - bi) for ai, bi in zip(exact, algorithm)])

    for d in ds.dlist:
        sor_values = [compute_error_sor(d, n, k) for n in range(2, ds.max_n)]
        help_line = [n for n in range(2, ds.max_n)]
        # lu_values = [compute_error_lu(d, n) for n in range(2, ds.max_n)]
        plt.loglog(range(2, ds.max_n), sor_values)
        plt.loglog(help_line, [0.05 for n in help_line])
        # plt.loglog(range(2, ds.max_n), lu_values)
    plt.show()

def main():
    # setting_a = cs(2, 15)
    # plot_omega(setting_a)
    setting_b = cs(2, 20) # matrix size about 262,000
    plot_omega(setting_b)

def main2():
    settings_1 = cs(2, 20) # very long, but doable
    # plot_omega(settings_1)
    settings_2 = cs(1, 1000) # long, doable, but boring
    # plot_omega(settings_2)
    settings_3 = cs(3, 10) # takes too long
    # plot_omega(settings_3)
    settings_4 = cs(3, 6) # not too long, best omega = 1.3
    # plot_omega(settings_4)
    # plot_absolute_error(settings_4.change_omega(1.3))
    test = cs()
    # test.change(omega=1.8, params=dict(eps=1e-8, max_iter=1000, min_red=0))
    print(test.params)
    # compare_lu_sor(test, ds().change(dlist=[1], max_n=50))
    # compare_lu_sor(test, ds().change(dlist=[1], max_n=30), 2)

if __name__ == "__main__":
    main()
