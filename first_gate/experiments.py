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
    # SET PARAMETERS HERE
    a, b = np.pi, 3 * np.pi # the start and end point of the interval
    p = 1000 # the number of grid points in the interval

    h_to_test = (np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 10) # the values for h for the function
                                                              # plot
    h_values = np.logspace(-9, 2, num=50) # the values for h for the error plot
    j_values_small = (1.0 / 100 , 1.0 / 10) # the values for small j
    j_values_big = (10, 100) # the values for large j


    # say hello to human
    print("Greetings! This module will verify the experiments of the protocol. See the protocol for more infomation.\n\n")

    # EXPERIMENT ONE
    # we want to draw the functions for 4 different values of h

    print("Section: Approximation's Approach")
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

    # constructs the object

    test_obj = da.FiniteDifference(1, g_j(1), dg_j(1), ddg_j(1))
    for new_h in h_to_test:
        test_obj.h = new_h
        test_obj.draw_functions(a, b, p)

    # EXPERIMENT TWO
    # we want to draw an useful error plot

    print("Section: Anatomy of Errors")
    print("We will draw the error plot.")

    test_obj.draw_errors(a, b, p, h_values)


    # EXPERIMENT THREE

    print("EXPERIMENT THREE")

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
