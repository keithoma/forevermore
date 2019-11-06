#!/usr/bin/python3

import numpy as np
from decimal import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

class AdditionalExperiments:
    """ This class's purpose is more to group methods together. It is used to draw the plots for the
    protocol.
    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    p : int
        Number of grid points in the interval.
    f : callable
        The function.
    df : callable
        The first analytic derivative of the function.
    ddf : callable
        The second analytic derivative of the function.
    defaultx : float
        The point where we will evaluate the approximation.
    Attributes
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    p : int
        Number of grid points in the interval.
    f : callable
        The function.
    df : callable
        The first analytic derivative of the function.
    ddf : callable
        The second analytic derivative of the function.
    defaultx : float
        The point where we will evaluate the approximation.
    """
    def __init__(self, a, b, p, f, df, ddf, defaultx):
        """ The constructer which just saves the arguments as attributes. See class docstring for more
        information.
        """
        self.a = a
        self.b = b
        self.p = p

        self.f = f
        self.df = df
        self.ddf = ddf
        self.defaultx = defaultx


    def approximate_first_derivative(self, x, h):
        return (self.f(x + h) - self.f(x)) / h

    def approximate_second_derivative(self, x, h):
        return (self.f(x + h) - 2 * self.f(x) + self.f(x - h)) / h ** 2

    def draw_first_approximation(self):

        # here, we compute the values for the plot
        h_values = np.logspace(-9, -6, num=500)
        d1_approx_values = [self.approximate_first_derivative(self.defaultx, h) for h in h_values]
        d1_analytic_values = [self.df(self.defaultx) for h in h_values]

        # draw the plot itself
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.loglog(h_values, d1_approx_values, label="$D_h^{(1)}f(" + str(round(self.defaultx, 2)) + ")$")
        plt.loglog(h_values, d1_analytic_values, label="$f'(" + str(round(self.defaultx, 2)) + ")$")

        # some plastic surgery
        plt.xlabel("h")
        plt.ylabel("$y$")
        plt.title("Plot of $D_h^{(1)}f$ and $f'$ at $x = " + str(round(self.defaultx, 2)) + " $ with $h$ as the variable")
        plt.legend()

        # we are working with very small numbers which means we need custom ticks
        # one may humor me and write this with np.arange
        ticks1 = (0.1153335, 0.11533351, 0.11533352, 0.11533353, 0.11533354, 0.11533355, 0.11533357)
        ticks2 = (0.11533358, 0.11533359, 0.1153336, 0.11533361, 0.11533362, 0.11533363)
        ticks = ticks1 + ticks2
        labels = (str(value) for value in ticks)
        plt.yticks(ticks, labels, fontsize=8)

        # finish
        plt.show()

    def draw_second_approximation(self):

        # calculate the values
        h_values = np.logspace(-5, -2, num=500)
        d2_approx_values = [self.approximate_second_derivative(self.defaultx, h) for h in h_values]
        d2_analytic_values = [self.ddf(self.defaultx) for h in h_values]

        # draw the plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.loglog(h_values, d2_approx_values, label="$D_h^{(1)}f(" + str(round(self.defaultx, 2)) + ")$")
        plt.loglog(h_values, d2_analytic_values, label="$f'(" + str(round(self.defaultx, 2)) + ")$")

        # legends and stuff
        plt.xlabel("h")
        plt.ylabel("$y$")
        plt.title("Plot of $D_h^{(1)}f$ and $f'$ at $x = " + str(round(self.defaultx, 2)) + " $ with $h$ as the variable")
        plt.legend()

        # we need custom ticks because we are woring with very small numbers
        # again, if one wants to arange this, be my guest
        ticks = (0.1319879, 0.131988, 0.1319881, 0.1319882, 0.1319883, 0.1319884)
        labels = (str(value) for value in ticks)
        plt.yticks(ticks, labels, fontsize=8)

        plt.show()

def main():
    """ The main function. It draws the plot for the protocol.
    """
    # some initial values
    a = np.pi
    b = 3 * np.pi
    p = 1000
    defaultx = np.pi + 2

    def f(x):
        """ A mathemathical function for testing.
        Parameters
        ----------
        x : float
            The variable.
        Returns
        -------
        float
            The solution of the function.
        """
        return np.sin(x) / x

    def df(x):
        """ The first derivative of f(x).
        Parameters
        ----------
        x : float
            The variable.
        Returns
        -------
        float
            The solution of the function.
        """
        numerator = x * np.cos(x) - np.sin(x)
        denominator = x ** 2
        return numerator / denominator

    def ddf(x):
        """ The second derivative of f(x).
        Parameters
        ----------
        x : float
            The variable.
        Returns
        -------
        float
            The solution of the function.
        """
        numerator = (x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)
        denominator = x ** 3
        return - numerator / denominator

    test_obj = AdditionalExperiments(a, b, p, f, df, ddf, defaultx)

    # demonstration
    print("Hello, this module draws the plot used for the accompanying protocol.")
    print()
    print("Consider f(x) = sin(x) / x.")
    print("We draw the plot for f' and its approximation for a fixed x and with h as the variable.")
    print()
    test_obj.draw_first_approximation()
    print("We draw the plot for f'' and its approximation for a fixed x and with h as the variable.")
    test_obj.draw_second_approximation()
    print()
    print("bye")

if __name__ == "__main__":
    main()
