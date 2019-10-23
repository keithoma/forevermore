#!/usr/bin/python3
"""
Authors: Christian Parpart () & Kei Thoma (574613)
Date:

ToDo:
* write doc string
* pylint
* create developer version and final version (that is delete some comments that are marked as such)

"""

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        The analytic first derivative of `f`.
    dd_f : callable, optional
        The analytic second derivative of `f`.
    Attributes
    ----------
    h : float
        Stepzise of the approximation.
    """



    def __init__(self, h, f, d_f=None, dd_f=None):
        # save parameters as attributes
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f



    def partition_interval_(self, interval, number_of_points):
        # if a > b, swap a and b
        if interval[0] > interval[1]:
            interval[0], interval[1] = interval[0], interval[1]
            print("Warning: Input interval [a, b] seems to have a > b.")

        # create partition
        dist = (interval[1] - interval[0]) / float(number_of_points)
        partition = [interval[0] + integer * dist for integer in range(0, number_of_points)]

        return partition



    def create_value_table_(self, _function, _partition):
        value_table = []
        for number in _partition:
            value_table.append(_function(number))
        return value_table



    def approximate_first_derivative(self, x):
        return (self.f(x + self.h) - self.f(x)) / self.h



    def approximate_second_derivative(self, x):
        return (self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)) / self.h ** 2



    def compute_errors(self, a, b, p): # pylint: disable=invalid-name
        # if of the analytic derivatives was not provided by the user, raise alarm
        if self.d_f == None or self.dd_f == None:
            raise ValueError("Not both analytic derivative was provided by the user.")

        # partition the interval [a, b]
        interval = (a, b)
        partition = self.partition_interval_(interval, p)

        # compute the values on these partition points
        d1_analytic_values = self.create_value_table_(self.d_f, partition)
        d1_approx_values = self.create_value_table_(self.approximate_first_derivative, partition)
        d2_analytic_values = self.create_value_table_(self.dd_f, partition)
        d2_approx_values = self.create_value_table_(self.approximate_second_derivative, partition)

        # finally, find the largest difference
        first_error  = max([abs(a_i - b_i) for a_i, b_i in zip(d1_analytic_values, d1_approx_values)])
        second_error = max([abs(a_i - b_i) for a_i, b_i in zip(d2_analytic_values, d2_approx_values)])

        # and return
        return first_error, second_error



    def draw_functions(self, a, b, p): # pylint: disable=invalid-name
        # partition the interval [a, b]
        interval = (a, b)
        partition = self.partition_interval_(interval, p)

        # compute the values on these partition points
        the_function_values = self.create_value_table_(self.f, partition)
        d1_approx_values = self.create_value_table_(self.approximate_first_derivative, partition)
        d2_approx_values = self.create_value_table_(self.approximate_second_derivative, partition)

        if self.d_f is not None:
            d1_analytic_values = self.create_value_table_(self.d_f, partition)

        if self.dd_f is not None:
            d2_analytic_values = self.create_value_table_(self.dd_f, partition)

        # draw the plot
        plt.figure(1)
        matplotlib.pyplot.grid(which="major")

        plt.plot(partition, the_function_values, label="f")

        if self.d_f is not None:
            plt.plot(partition, d1_analytic_values, label="analytic f'")

        plt.plot(partition, d1_approx_values, label="approximation of f'", linestyle='dashed')

        if self.dd_f is not None:
            plt.plot(partition, d2_analytic_values, label="analytic f''")

        plt.plot(partition, d2_approx_values, label="approximation of f''", linestyle='dashed')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Plot of f And Its Derivatives")
        plt.legend()

        plt.show()



    def draw_errors(self, a, b, p, h_values):
        # partition the interval [a, b]
        interval = (a, b)
        partition = self.partition_interval_(interval, p)

        # we will iterate over each value in h_values to find the errors of the analytic and its
        # approximation
        # this could have been done with list comprehension, but if done so, compute errors would have
        # needed another argument for h
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
        matplotlib.pyplot.grid(which="major")

        plt.loglog(h_values, h_values, label="h")
        plt.loglog(h_values, h_quadratic, label="h^2")
        plt.loglog(h_values, h_cubic, label="h^3")

        plt.loglog(h_values, d1_error_values, label="d1 error")
        plt.loglog(h_values, d2_error_values, label="d2 error")

        plt.xlabel("values of h")
        plt.ylabel("error")
        plt.title("Plot of the Errors")
        plt.legend()

        plt.show()



def main():
    """
    The plot drawn with Desmos:

    """
    # greetings
    print("This module is about approximations of derivatives!")
    print("Consider the function g_1(x) = sin(x) / x\n\n")

    # to test our module, we will consider the function defined below
    def g_1(x):
        return np.sin(x) / x

    # the following two functions are the first two analytic derivatives
    def dg_1(x):
        numerator = x * np.cos(x) - np.sin(x)
        denominator = x ** 2
        return numerator / denominator

    def ddg_1(x):
        numerator = (x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)
        denominator = x ** 3
        return - numerator / denominator

    # some initial values for testing
    a, b = np.pi, 3 * np.pi
    p = 1000
    h_values = np.logspace(-9, 2, num = 50)

    # construct an object
    initial_h = 0.5
    # initial_h = 1 # developer version only
    test_class = FiniteDifference(initial_h, g_1, dg_1, ddg_1)

    # test_class = FiniteDifference(initial_h, g_1) # developer version only

    # demonstration of the approximations
    # if x = 5, then it should be
    #
    #                        analytic | approximated
    # first derivative    |  0.0950   | 0.1270
    # second derivative   |  0.1537   | 0.1522
    x = 5
    d1_solution = test_class.approximate_first_derivative(x)
    d2_solution = test_class.approximate_second_derivative(x)
    print("Firstly, this module is able to compute the approximation of the derivatives of a given " +
          "function.")
    print("The first derivative of g_1 at x = {0} should be about {1} (exact: {2}).".format(x, d1_solution, dg_1(x)))
    print("The second derivative of g_1 at x = {0} should be about {1} (exact: {2}).".format(x, d2_solution, ddg_1(x)))
    print("\n")

    # demonstration of the error calculations
    # check the output values with the data here:
    # https://docs.google.com/spreadsheets/d/1aeaQkveMq4S_MCfeT2-_4mMtH0bFrduJscciTH_FwIw/edit?usp=sharing
    if test_class.d_f is not None and test_class.dd_f is not None:
        first_error, second_error = test_class.compute_errors(a, b, p)
        print("Now, consider the interval [{0}, {1}] partitioned into {2} many points.".format(a, b, p))
        print("Then, the maximal difference of the analytic derivative and the approximated functions are,")
        print("{0} for the first derivative, and".format(first_error))
        print("{0} for the second derivative.".format(second_error))
        print("\n")

    # demonstration of the plotting feature
    test_class.draw_functions(a, b, p)
    print("We can also draw the plot of the given function and its derivatives.")
    print("\n")

    # demonstration of the plotting of the errors
    if test_class.d_f is not None and test_class.dd_f is not None:
        test_class.draw_errors(a, b, p, h_values)
        print("Furthermore, we can also plot the errors of the approximation and the analytic derivatives.")
        print("\n")

    print("Stay classy!")
    print("\n")
    print("END OF DEMONSTRATION\n")

if __name__ == "__main__":
    main()
