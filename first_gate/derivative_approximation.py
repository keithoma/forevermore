"""
Author: Christian Parpart & Kei Thoma
Date:

Naming Convention:
* prefix underscore denotes parameters
* subfix underscore denotes private functions

"""



import math

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

        # initialize the graphics
        self.graphics = {'fig': None,
                         'ax_f': None,
                         'ax_d1': None,
                         'ax_d2': None,
                         'errfig': None,
                         'ax_err': None}

    def partition_interval_(self, _interval, _number_of_points):
        # todo: what happens if a = b?

        # if a > b, swap a and b
        if _interval[0] > _interval[1]:
            _interval[0], _interval[1] = _interval[0], _interval[1]

        # create partition
        partition = []
        dist = (_interval[1] - _interval[0]) / _number_of_points
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

    def draw_functions(self, interval_, number_of_points_):
        pass



def main():
    """ Presents a quick example ... (TODO) ...
    """
    def a_function(x):
        return math.log(x)

    def derivative(x):
        return 1 / x

    def second_derivative(x):
        return -1 / (2 * x ** 2)

    a_class = FiniteDifference(0.01, a_function, derivative, second_derivative)
    print(a_class.compute_errors((1, 11), 11))


if __name__ == "__main__":
    main()
