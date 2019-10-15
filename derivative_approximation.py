"""

Author: Christian Parpart & Kei Thoma
Date:
"""

import math

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

    def calculate_derivative(self, x):
        return (self.f(x + self.h) - self.f(x)) / self.h

    def calculate_second_derivative(self, x):
        return self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)

    def compute_errors(self, a, b, p):  # pylint: disable=invalid-name
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

        # TODO: what happens if p is 0 or negative

        # if a is larger than b, swap a and b
        if a > b:
            a, b = b, a

        # TODO: what happens if a = b?

        # we will fill this list with points we want to consider
        list_of_points = []
        # for that, we need the distance between each point
        dis = (b - a) / p
        # now fill the list
        for i in range(0, p):
            list_of_points.append(a + (i * dis))

        def max_error(analytic_f, approximated_f, _list_of_points):
            error_list = []
            for value in _list_of_points:
                error = abs(analytic_f(value) - approximated_f(value))
                error_list.append(error)
            return max(error_list)

        first_error  = max_error(self.d_f, self.calculate_derivative, list_of_points)
        second_error = max_error(self.dd_f, self.calculate_second_derivative, list_of_points)
        return first_error, second_error



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
