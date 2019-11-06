#!/usr/bin/python3

import numpy as np
from decimal import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

class AdditionalExperiments:
    def __init__(self, a_, b_, p_, f_, df_, ddf_, defaultx_):
        self.a = a_
        self.b = b_
        self.p = p_

        self.f = f_
        self.df = df_
        self.ddf = ddf_
        self.defaultx = defaultx_


    def approximate_first_derivative(self, x, h):
        return (self.f(x + h) - self.f(x)) / h

    def approximate_second_derivative(self, x, h):
        return (self.f(x + h) - 2 * self.f(x) + self.f(x - h)) / h ** 2

    def draw_first_approximation(self):
        h_values = np.logspace(-9, -6, num=500)
        d1_approx_values = [self.approximate_first_derivative(self.defaultx, h) for h in h_values]
        d1_analytic_values = [self.df(self.defaultx) for h in h_values]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)


        plt.loglog(h_values, d1_approx_values, label="$D_h^{(1)}f(" + str(round(self.defaultx, 2)) + ")$")
        plt.loglog(h_values, d1_analytic_values, label="$f'(" + str(round(self.defaultx, 2)) + ")$")

        plt.xlabel("h")
        plt.ylabel("$y$")
        plt.title("Plot of $D_h^{(1)}f$ and $f'$ at $x = " + str(round(self.defaultx, 2)) + " $ with $h$ as the variable")
        plt.legend()
        ticks = [0.11533351, 0.1153336]

        ax.set_yticks(ticks)
        plt.show()

    def draw_second_approximation(self):
        h_values = np.logspace(-5, -2, num=500)
        d2_approx_values = [self.approximate_second_derivative(self.defaultx, h) for h in h_values]
        d2_analytic_values = [self.ddf(self.defaultx) for h in h_values]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)


        plt.loglog(h_values, d2_approx_values, label="$D_h^{(1)}f(" + str(round(self.defaultx, 2)) + ")$")
        plt.loglog(h_values, d2_analytic_values, label="$f'(" + str(round(self.defaultx, 2)) + ")$")

        plt.xlabel("h")
        plt.ylabel("$y$")
        plt.title("Plot of $D_h^{(1)}f$ and $f'$ at $x = " + str(round(self.defaultx, 2)) + " $ with $h$ as the variable")
        plt.legend()
        # ticks = [0.11533351, 0.1153336]

        # ax.set_yticks(ticks)
        plt.show()

    def draw_first_difference(self):
        h_values = np.logspace(-17, -14, num=10000)
        d1_difference_values = [self.f(self.defaultx + h) - self.f(self.defaultx) for h in h_values]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.loglog(h_values, d1_difference_values)

        plt.show()

def main():
    a = np.pi
    b = 3 * np.pi
    p = 1000

    def f(x):
        return np.sin(x) / x

    def df(x):
        numerator = x * np.cos(x) - np.sin(x)
        denominator = x ** 2
        return numerator / denominator

    def ddf(x):
        numerator = (x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)
        denominator = x ** 3
        return - numerator / denominator

    defaultx = np.pi + 2

    test_obj = AdditionalExperiments(a, b, p, f, df, ddf, defaultx)
    test_obj.draw_first_approximation()
    test_obj.draw_second_approximation()
    # test_obj.draw_first_difference()

if __name__ == "__main__":
    main()
