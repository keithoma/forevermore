#!/usr/bin/python3

import numpy as np

import derivative_approximation as da

def main():
    # say hello to human
    print("Greetings! This module will verify the experiments of the protocol. See the protocol for more infomation.\n\n")

    # EXPERIMENT ONE
    # we want to draw the functions for 4 different values of h

    print("EXPERIMENT ONE")
    print("We will draw the plot for pi/3, pi/4, pi/5 and pi/10.\n\n")

    # consider the function given below
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

    # the following two functions are the first two analytic derivatives of g_1
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

    # initialize some constants and construct an object of FiniteDifference
    h_to_test = (np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 10)
    test_obj = da.FiniteDifference(1, g_1, dg_1, ddg_1)
    a, b = np.pi, 3 * np.pi
    p = 1000

    # draw the graphs for each h
    if False:
        for new_h in h_to_test:
            test_obj.h = new_h
            test_obj.draw_functions(a, b, p)

    # EXPERIMENT TWO
    # we want to draw an useful error plot

    print("EXPERIMENT TWO")
    print("We will draw the error plot.")

    h_values = np.logspace(-9, 2, num=50)
    test_obj.draw_errors(a, b, p, h_values)



if __name__ == "__main__":
    main()
