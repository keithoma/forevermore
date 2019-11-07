#!/usr/bin/python3
"""
Authors: Christian Parpart (185 676) & Kei Thoma (574 613)
Date: 24th of October 2019

This module implements a class FiniteDifference which provides functions to compute the approximation of
a given function's derivatives. It is also able to draw a corresponding plot and calculate the diviation of
the approximation from the analytic derivatives if they were provided by the user.
"""

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position

class FiniteDifference:
    """ Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.
    Parameters
    ----------
    h : float
        Stepzise of the approximation.
    f : callable
        Function to approximate the derivatives of.
    d_f : callable, optional
        The analytic first derivative of 'f'.
    dd_f : callable, optional
        The analytic second derivative of 'f'.
    Attributes
    ----------
    h : float
        Stepzise of the approximation.
    f : callable
        Function to approximate the derivatives of.
    d_f : callable or None
        The analytic first derivative of 'f'.
    dd_f : callable or None
        The analytic second derivative of 'f'.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        """ The constructor saves the arguments passed directly as attributes. See the class docstring for
        a description for the parameters and attributes.
        """
        # save parameters as attributes
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f

        # graphic settings
        self.error_plot_range = 10 ** (-9), 10 ** 5
        self.j = 1

    @staticmethod
    def partition_interval_(a, b, p):
        """ This is a (private) function used to partition a interval passed as a parameter.
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            The number of points in this interval, a and b are counted.
        Returns
        -------
        list
            A list of floats which contains p number of points separated evenly between a and b including a
            and b.
        """
        # if a > b, swap a and b
        if a > b:
            a, b = b, a
            print("Warning: Input interval [a, b] seems to have a > b.")

        # create partition
        dist = (b - a) / float(p)
        partition = [a + integer * dist for integer in range(0, p)]

        return partition

    def approximate_first_derivative(self, x):
        """ This function approximates the first derivative of f at a given point. The formula used is the
        the courtesy of the Taylor's theorem.
        Parameters
        ----------
        x : float
            the point where the first derivative of f should be approximated
        Returns
        -------
        float
            the approximation of the first derivative at the given point x
        """
        return (self.f(x + self.h) - self.f(x)) / self.h

    def approximate_second_derivative(self, x):
        """ This function approximates the second derivative of f at a given point. The formula used is the
        the courtesy of the Taylor's theorem.
        Parameters
        ----------
        x : float
            the point where the second derivative of f should be approximated
        Returns
        -------
        float
            the approximation of the second derivative at the given point x
        """
        return (self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)) / self.h ** 2

    def compute_errors(self, a, b, p):
        """ Calculates an approximation to the errors between an approximation
        and the exact derivative for first and second order derivatives in the
        maximum norm.
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            Number of points used in the approximation of the maximum norm.
        Returns
        -------
        float
            Errors of the approximation of the first derivative.
        float
            Errors of the approximation of the second derivative.
        Raises
        ------
        ValueError
            If no analytic derivative was provided by the user.
        """
        # if of the analytic derivatives was not provided by the user, raise alarm
        if self.d_f is None or self.dd_f is None:
            raise ValueError("Not both analytic derivative was provided by the user.")

        # partition the interval [a, b]
        partition = self.partition_interval_(a, b, p)

        # compute the values on these partition points
        d1_analytic_values = [self.d_f(number) for number in partition]
        d1_approx_values = [self.approximate_first_derivative(number) for number in partition]
        d2_analytic_values = [self.dd_f(number) for number in partition]
        d2_approx_values = [self.approximate_second_derivative(number) for number in partition]

        # finally, find the largest difference
        first_error = max([abs(a_i - b_i) for a_i, b_i in zip(d1_analytic_values, d1_approx_values)])
        second_error = max([abs(a_i - b_i) for a_i, b_i in zip(d2_analytic_values, d2_approx_values)])

        # and return
        return first_error, second_error

    def draw_functions(self, a, b, p):
        """ This function draws the plot for f, the approximated first two derivatives and if applicable,
        the analytic first two derivatives provided by the user.
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            The number of points in this interval, a and b are counted.
        """
        # partition the interval [a, b]
        partition = self.partition_interval_(a, b, p)

        # compute the values on these partition points
        the_function_values = [self.f(number) for number in partition]
        d1_approx_values = [self.approximate_first_derivative(number) for number in partition]
        d2_approx_values = [self.approximate_second_derivative(number) for number in partition]

        if self.d_f is not None:
            d1_analytic_values = [self.d_f(number) for number in partition]

        if self.dd_f is not None:
            d2_analytic_values = [self.dd_f(number) for number in partition]

        # draw the plot
        plt.figure(1)
        matplotlib.pyplot.grid(which="major")

        plt.plot(partition, the_function_values, label="$f$")

        if self.d_f is not None:
            plt.plot(partition, d1_analytic_values, label="$f'$")

        plt.plot(partition, d1_approx_values, label="$D^{(1)}_h(f)$", linestyle='dashed')

        if self.dd_f is not None:
            plt.plot(partition, d2_analytic_values, label="$f''$")

        plt.plot(partition, d2_approx_values, label="$D^{(2)}_h(f)$", linestyle='dashed')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Plot of f and Its Derivatives (h = {})".format(round(self.h, 2)))
        plt.legend()

        plt.show()

    def draw_errors(self, a, b, p, h_values):
        """ This function draws the error plot (i.e. the maximum difference of the analytic and the
        approximated on the given interval) according to the values for h with the double log scale. In
        addition, the plots for h, h^2, and h^3 are also drawn.
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            The number of points in this interval, a and b are counted.
        h_values : list
            This list should contain the values for h for which the error plot is drawn.
        """
        # again raise alarm if analytic derivatives were not provided
        if self.d_f is None or self.dd_f is None:
            raise ValueError("Not both analytic derivative was provided by the user.")

        # we will iterate over each value in h_values to find the errors of the analytic and its
        # approximation
        reset_h = self.h
        d1_error_values = []
        d2_error_values = []

        for new_h in h_values:
            self.h = new_h
            d1_error_values.append(self.compute_errors(a, b, p)[0])
            d2_error_values.append(self.compute_errors(a, b, p)[1])

        self.h = reset_h

        # we will also compute the values for h ** 2 and h ** 3
        h_quadratic = [h ** 2 for h in h_values]
        h_cubic = [h ** 3 for h in h_values]

        # draw the plot
        plt.figure(2)
        error_ax = plt.gca()
        matplotlib.pyplot.grid(which="major")

        plt.loglog(h_values, h_values, label="$h$", linestyle='dashed')
        plt.loglog(h_values, h_quadratic, label="$h^2$", linestyle='dashed')
        plt.loglog(h_values, h_cubic, label="$h^3$", linestyle='dashed')

        plt.loglog(h_values, d1_error_values, label="$e^{(1)}_f$")
        plt.loglog(h_values, d2_error_values, label="$e^{(2)}_f$")

        error_ax.set_ylim(self.error_plot_range) # limit the range of y; the values outside are uninteresting
                                                 # for our purposes
        plt.xlabel("h")
        plt.ylabel("error or $h^j$")
        plt.title("Plot of $e^{i}_f$ and $h^j$ (j = " + str(round(self.j, 2)) + ")") # because format() gets confused
        plt.legend()

        plt.show()

    def set_error_range(self, bottom, top):
        self.error_plot_range = bottom, top

    def set_j(self, j):
        self.j = j

def main():
    """ Our glorious main function. It will demonstrate every feature implemented in this module for the
    function, sin(x) / x.
    """
    # greetings
    print("This module is about approximations of derivatives!")
    print("Consider the function g_1(x) = sin(x) / x\n\n")

    # to test our module, we will consider the function defined below
    def g_1(x):
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

    # the following two functions are the first two analytic derivatives
    def dg_1(x):
        """ The first derivative of g_1().
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

    def ddg_1(x):
        """ The second derivative of g_1().
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

    # some initial values for testing
    a, b = np.pi, 3 * np.pi
    p = 1000
    h_values = np.logspace(-9, 2, num=50)

    # construct an object
    initial_h = 0.5
    test_obj = FiniteDifference(initial_h, g_1, dg_1, ddg_1)

    # demonstration of the approximations
    x = 5
    d1_solution = test_obj.approximate_first_derivative(x)
    d2_solution = test_obj.approximate_second_derivative(x)
    print("Firstly, this module is able to compute the approximation of the derivatives of a given " +
          "function.")
    print("The first derivative of g_1 at x = {0} should be about {1} (exact: {2}).".format(x, d1_solution, dg_1(x)))
    print("The second derivative of g_1 at x = {0} should be about {1} (exact: {2}).".format(x, d2_solution, ddg_1(x)))
    print("\n")

    # demonstration of the error calculations
    if test_obj.d_f is not None and test_obj.dd_f is not None:
        first_error, second_error = test_obj.compute_errors(a, b, p)
        print("Now, consider the interval [{0}, {1}] partitioned into {2} many points.".format(a, b, p))
        print("Then, the maximal difference of the analytic derivative and the approximated functions are,")
        print("{0} for the first derivative, and".format(first_error))
        print("{0} for the second derivative.".format(second_error))
        print("\n")

    # demonstration of the plotting feature
    # Also check the plot drawn with Desmos:
    #      https://www.desmos.com/calculator/eiacy1i3nk
    print("We can also draw the plot of the given function and its derivatives.")
    print("\n")
    test_obj.draw_functions(a, b, p)

    # demonstration of the plotting of the errors
    if test_obj.d_f is not None and test_obj.dd_f is not None:
        print("Furthermore, we can also plot the errors of the approximation and the analytic derivatives.")
        print("\n")
        test_obj.draw_errors(a, b, p, h_values)

    print("Stay classy!")
    print("\n")
    print("END OF DEMONSTRATION\n")

if __name__ == "__main__":
    main()
