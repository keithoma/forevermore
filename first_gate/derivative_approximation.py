"""

Author: Christian Parpart & Kei Thoma
Date:

Fragen:
* duerfen wir den vorgegebenen Quelltext veraendern? (nicht, dass wir das direkt vorhaben, aber ich will
  nicht unbedingt spaeter nochmal durchschauen ob auch nichts veraendert wurde)
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

    def _partition_interval(interval_, number_of_points_):
        # todo: what happens if a = b?

        # if a > b, swap a and b
        if interval_[0] > interval_[1]:
            interval_[0], interval_[1] = interval_[0], interval_[1]
        
        # create partition
        partition = []
        dist = (interval_[1] - interval_[0]) / number_of_points_
        for integer in range(0, number_of_points_):
            partition.append(interval_[0] + integer * dist)
        
        return partition

    def _create_value_table(function_, partition_):
        value_table = []
        for number in partition_:
            value_table.append(function_(number))
        return value_table

    def approximate_first_derivative(self, x):
        return (self.f(x + self.h) - self.f(x)) / self.h

    def approximate_second_derivative(self, x):
        return self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)

    def compute_errors(self, interval_, number_of_points_):  # pylint: disable=invalid-name
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
        # if neither analytic derivative was provided by the user, raise alarm
        # note that if either one of the derivative was provided, then we will proceed normally
        if self.d_f == None and self.dd_f == None:
            raise ValueError("No analytic derivative was provided by the user.")

        partition = self.partition_interval()
    
        d1_analytic_values = self._create_value_table(self.d_f, partition)
        d1_approx_values = self._create_value_table(self.approximate_first_derivative, partition)
        d2_analytic_values = self._create_value_table(self.dd_f, partition)
        d2_approx_values = self._create_value_table(self.approximate_second_derivative, partition)

        first_error  = max(abs(d1_analytic_values - d1_approx_values))
        second_error = max(abs(d2_analytic_values - d2_approx_values))
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
    print(a_class.compute_errors(1, 11, 11))


if __name__ == "__main__":
    main()
