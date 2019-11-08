#!/usr/bin/python3

import numpy as np
import derivative_approximation as da

def main():
    """
    This is the main function being invoked when running this program.

    Parameter:
    ----------
        None

    Returns:
    --------
        None
    """
    # say hello to human
    print("Greetings! This module will verify the experiments of the protocol. See the protocol for more infomation.\n\n")

    # EXPERIMENT ONE
    # we want to draw the functions for 4 different values of h

    print("EXPERIMENT ONE")
    print("We will draw the plot for pi/3, pi/4, pi/5 and pi/10.\n\n")

    # consider the function given below
    def g_j(j):
        """
        Constructs A mathemathical function for testing sin(jx)/x.

        Parameters
        ----------
        j : float
            The j-variable inside the mathematical expression sin(jx)/x.

        Returns
        -------
        callable
            The mathematical function sin(jx)/x with fixiated j.
        """
        return lambda x: np.sin(j * x) / x

    # the following two functions are the first two analytic derivatives of g_1
    def dg_j(j):
        """
        Constructs the first derivative of g_1() with fixiated j.

        Parameters
        ----------
        j : float
            The variable.

        Returns
        -------
        float
            The solution of the function.
        """
        numerator = lambda x: j * x * np.cos(j * x) - np.sin(j * x)
        denominator = lambda x: x ** 2
        return lambda x: numerator(x) / denominator(x)

    def ddg_j(j):
        """
        The second derivative of g_1().

        Parameters
        ----------
        j : float
            The variable.

        Returns
        -------
        float
            The solution of the function.
        """
        numerator = lambda x: (2 - (j ** 2) * (x ** 2)) * np.sin(j * x) - 2 * j * x * np.cos(j * x)
        denominator = lambda x: x ** 3
        return lambda x: numerator(x) / denominator(x)

    # initialize some constants and construct an object of FiniteDifference
    a, b = np.pi, 3 * np.pi
    p = 1000

    # EXPERIMENT TWO
    # we want to draw an useful error plot

    print("EXPERIMENT TWO")
    print("We will draw the error plot.")

    h_values = np.logspace(-9, 2, num=50)

    # EXPERIMENT THREE

    print("EXPERIMENT THREE")

    j_values_small = (1.0 / 100 , 1.0 / 10)
    j_values_big = (10, 100)

    for j in j_values_small:
        j_obj = da.FiniteDifference(1, g_j(j), dg_j(j), ddg_j(j))
        j_obj.ghost_error_plot = True
        j_obj.set_error_range(10 ** (-15), 10 ** 5)
        j_obj.set_j(j)
        j_obj.draw_errors(a, b, p, h_values)

    for j in j_values_big:
        j_obj = da.FiniteDifference(1, g_j(j), dg_j(j), ddg_j(j))
        j_obj.ghost_error_plot = True
        j_obj.set_error_range(10 ** (-15), 10 ** 5)
        j_obj.set_j(j)
        j_obj.draw_errors(a, b, p, h_values)

if __name__ == "__main__":
    main()
