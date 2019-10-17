#!/usr/bin/python3
"""
Author: Christian Parpart & Kei Thoma
Date:

Naming Convention:
* prefix underscore denotes parameters
* subfix underscore denotes private functions

# TODO:
* missing the plots of h -> h; h -> h^2 and h -> h^3

"""



import math
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

        matplotlib.pyplot.grid(which="major")

    def partition_interval_(self, _interval, _number_of_points):
        # todo: what happens if a = b?

        # if a > b, swap a and b
        if _interval[0] > _interval[1]:
            _interval[0], _interval[1] = _interval[0], _interval[1]

        # create partition
        partition = []
        dist = (_interval[1] - _interval[0]) / float(_number_of_points)
        for integer in range(0, _number_of_points):
            partition.append(_interval[0] + integer * dist)

        return partition

    def create_value_table_(self, _function, _partition):
        value_table = []
        for number in _partition:
            value_table.append(_function(number))
        return value_table

    def approximate_first_derivative(self, x):
        return (self.f(x + self.h) - self.f(x)) / self.h

    def approximate_second_derivative(self, x):
        return self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)

    def compute_errors(self, _interval, _number_of_points):  # pylint: disable=invalid-name
        # if of the analytic derivatives was not provided by the user, raise alarm
        if self.d_f == None or self.dd_f == None:
            raise ValueError("Not both analytic derivative was provided by the user.")

        partition = self.partition_interval_(_interval, _number_of_points)

        d1_analytic_values = self.create_value_table_(self.d_f, partition)
        d1_approx_values = self.create_value_table_(self.approximate_first_derivative, partition)
        d2_analytic_values = self.create_value_table_(self.dd_f, partition)
        d2_approx_values = self.create_value_table_(self.approximate_second_derivative, partition)

        first_error  = max([abs(a_i - b_i) for a_i, b_i in zip(d1_analytic_values, d1_approx_values)])
        second_error = max([abs(a_i - b_i) for a_i, b_i in zip(d2_analytic_values, d2_approx_values)])
        return first_error, second_error

    def draw_functions(self, _interval, _number_of_points):
        # create a private function for the following 5 lines TODO
        partition = self.partition_interval_(_interval, _number_of_points)

        the_function_values = self.create_value_table_(self.f, partition)

        d1_analytic_values = self.create_value_table_(self.d_f, partition)
        d1_approx_values = self.create_value_table_(self.approximate_first_derivative, partition)
        d2_analytic_values = self.create_value_table_(self.dd_f, partition)
        d2_approx_values = self.create_value_table_(self.approximate_second_derivative, partition)

        plt.figure(1)

        plt.plot(partition, the_function_values, label="f")
        plt.plot(partition, d1_analytic_values, label="analytic f'")
        plt.plot(partition, d1_approx_values, label="approximation of f'", linestyle='dashed')
        plt.plot(partition, d2_analytic_values, label="analytic f''")
        plt.plot(partition, d2_approx_values, label="approximation of f''", linestyle='dashed')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Plot of f And Its Derivatives")
        plt.legend()

        plt.show()

    def draw_errors(self, _interval, _number_of_points, _h_values):
        # double log scale
        partition = self.partition_interval_(_interval, _number_of_points)

        d1_error_values = []
        d2_error_values = []
        for new_h in _h_values:
            self.h = new_h
            d1_error_values.append(self.compute_errors(_interval, _number_of_points)[0])
            d2_error_values.append(self.compute_errors(_interval, _number_of_points)[1])

        plt.figure(2)

        plt.loglog(_h_values, d1_error_values, label="d1 error")
        plt.loglog(_h_values, d2_error_values, label="d2 error")

        plt.xlabel("values of h")
        plt.ylabel("error")
        plt.title("Plot of the Errors")
        plt.legend()

        plt.show()

"""
def main():
    def a_function(x):
        return math.log(1 + x)

    def derivative(x):
        return 1 / (x + 1)

    def second_derivative(x):
        return -1 / (1 + x) ** 2

    a_class = FiniteDifference(0.01, a_function, derivative, second_derivative)
    print(a_class.compute_errors((2, 12), 10))
    #a_class.draw_functions((1, 10), 3000)

    h_values = np.arange(0.0001, 2, 0.0001)
    print(h_values)
    a_class.draw_errors((2, 12), 10, h_values)
"""

def main():
    def g_1(x):
        return math.sin(x) / x

    def dg_1(x):
        return (x * math.cos(x) - math.sin(x)) / x ** 2

    def ddg_1(x):
        return ((x ** 2 - 2) * math.sin(x) + 2 * x * math.cos(x)) / x ** 3

    number_of_points = 3000
    h_values = np.arange(0.001, 1, 0.001)

    test_class = FiniteDifference(0.5, g_1, dg_1, ddg_1)
    test_class.draw_functions((0.1, 20), number_of_points)
    # runtime warning below
    test_class.draw_errors((0.1, 20), number_of_points, h_values)

if __name__ == "__main__":
    main()
