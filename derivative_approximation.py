"""

Author: Christian Parpart & Kei Thoma
Date:
"""


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

        # we will fill this list with points we want to consider
        list_of_points = []
        # for that, we need the distance between each point
        dis = abs(a - b) / p
        # now fill the list
        # TODO: does not consider if a > b
        for i in range(0, p):
            list_of_points.append(a + dis * i)
        def max_error(analytic_f, approximated_f, a, b, p):
            pass



def main():
    """ Presents a quick example ... (TODO) ...

    """
    def a_function(x):
        return x ** 2
    a_class = FiniteDifference(0.01, a_function)
    print(a_class.calculate_derivative(5))


if __name__ == "__main__":
    main()
