#!/usr/bin/python3

"""
SERIE 1:
  Gegeben:
    - Intervall I := [a,b] mit reellem Intervall I
    - reelle Fkt. f: I -> R   (with R being set of real values)
    - erste Abl. f'
    - zweite Abl. f''
    - Schrittweite h der Approximation der Ableitungen
    - Anzahl p der Punkte des Plots
  Gesucht:
    - grafische Darstellung der Funktion f, den Ableitungen f' und f'',
      wobei diese drei als Funktionen uebergeben werden
    - grafische darstellung der Approximierten Ableiutungen (Sekanten statt Tangenten)
    - Fehlerberechnung: Abweichung zwischen echter und approximierter Ableitung

Autoren: Oliver Bandel (556 648), Kei Thoma (574 613)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D # needed for custom legend


MINIMUM_PLOTPOINTS = 2


class Approximation:
    """
    Approximation of first and second derivative (Working-task 1.1).

    :type intervall: [int,int]
    :param intervall: the intervall that will be plotted

    :type thefun: float -> float
    :param thefun: mathematical function

    :type deriv_1: float -> float
    :param deriv_1: mathematical function, 1st derivative of thefun

    :type deriv_2: float -> float
    :param deriv_2: mathematical function, 2nd derivative of thefun

    :type num_points: int
    :param num_points: number of plot-points (including the start and end of the intervall)

    __init__ stores the parameters as attributes and therafter caclulates
    the data sets, which contain f(x), f'(x), f''(x).
    Here the derivatives are calculated from the functions, given as
    parameters. So, they calculate the correct (non-approximated) values.

    After that, __init__ initializes the graphics for the function-plots.


    To access the data, you can access via
    ``obj.data[<name>]`` where ``name`` is one of

        - ``plotting_points`` (for the plotting-points (x-values),
        - ``funktionswerte`` (for the f(x)-values),
        - ``ableitung_1_theoretisch`` (for the  values of f'(x), theoretical)
        - ``ableitung_2_theoretisch`` (for the values of f''(x), theoretical)
        - ``d1_approx`` (for values of approximation of f'(x))
        - ``d2_approx`` (for values of approximation of f''(x))
        - ``h_values`` (for h-values (dist-values), used in calculation of error of approximation)

    """

    def __init__(self, intervall, thefun, deriv_1, deriv_2, num_points):
        """
        Initialisation with the following parmeters (described above):

        :type intervall: [int,int]
        :param intervall: the intervall that will be plotted

        :type thefun: float -> float
        :param thefun: mathematical function

        :type deriv_1: float -> float
        :param deriv_1: mathematical function, 1st derivative of thefun

        :type deriv_2: float -> float
        :param deriv_2: mathematical function, 2nd derivative of thefun

        :type num_points: int
        :param num_points: number of plot-points (including the start and end of the intervall)

        __init__ stores the parameters as attributes and therafter caclulates
        the data sets, which contain f(x), f'(x), f''(x).
        Here the derivatives are calculated from the functions, given as
        parameters. So, they calculate the correct (non-approximated) values.

        After that, __init__ initializes the graphics for the function-plots.

        To access the data, you can access via
        ``obj.data[<name>]`` where ``name`` is one of
            -``plotting_points``,
            -``funktionswerte``,
            -``ableitung_1_theoretisch``
            -``ableitung_2_theoretisch``
            -``d1_approx``
            -``d2_approx``
            -``h_values``
        """

        # Save parameters as attributes
        self.intervall_begin = intervall[0] #: intervall begin (float)
        self.intervall_end = intervall[1] #: intervall end (float)
        self.function = thefun #: function (math function)
        self.deriv_1 = deriv_1 #: deriv_1 (math function)
        self.deriv_2 = deriv_2 #: deriv_2 (math function)
        self.num_points = MINIMUM_PLOTPOINTS # minimum number of plot-points

        self.data = {'plotting_points': None,
                     'funktionswerte': None,
                     'ableitung_1_theoretisch': None,
                     'ableitung_2_theoretisch': None,
                     'd1_approx': None,
                     'd2_approx': None,
                     'h_values': np.array([])}

        self.set_plotpoints_and_calculate_theoretical_values(num_points)

        self.graphics = {'fig': None,
                         'ax_f': None,
                         'ax_d1': None,
                         'ax_d2': None,
                         'errfig': None,
                         'ax_err': None}




    ############ INITIAL CALCULATING FUNCTION ############

    def set_plotpoints_and_calculate_theoretical_values(self, num_points):
        """
        Set the number of plotpoints.
        Also calculate the plotpoints, function values and values for the theoretical 1st and 2nd derivation.

        :type num_points: int
        :param num_points: number of plot-points (including the start and end of the intervall)
        """

        # Create data
        # -----------
        # Check number of plot-points. Forcing minimum value.
        self.num_points = num_points
        if self.num_points < MINIMUM_PLOTPOINTS:
            self.num_points = MINIMUM_PLOTPOINTS

        # Create Numpy-Arrays with the Plotting-Points and the values for f(x), f'(x), f''(x)
        plotpts = np.linspace(self.intervall_begin, self.intervall_end, self.num_points, endpoint=True)
        fktw = self.function(plotpts)
        abl_1 = self.deriv_1(plotpts)
        abl_2 = self.deriv_2(plotpts)

        # Save data as attributes
        self.data['plotting_points'] = plotpts
        self.data['funktionswerte'] = fktw
        self.data['ableitung_1_theoretisch'] = abl_1
        self.data['ableitung_2_theoretisch'] = abl_2



    ############ CALCULATING FUNCTIONS ############

    @staticmethod
    def calc_difference_quotient(x_values, func, dist):
        """
        Calulating difference quotient (1st and second order) for an np.array of plotpoints/function-values.

        :type x_values: float array (python or np.array)
        :param x_values: x-values (at) the plotpoints

        :type func: mathematical function (e.g. math.sin)
        :param func: function, which will be analysed (see task description about matjematical propertries)

        :type dist: float
        :param dist: Value for 'h' in task description


        ( f(x+dist) - f(x) ) / h
        and
        (f(x+dist - f*f(x) + f(x-h)) / h^2.

        **Returns: [deriv_1_approx, deriv_2_approx]** list of difference-quotient arrays
        """

        dist = abs(dist) # absolute value, just in case someone uses negative values

        # some calculations beforehand
        x_minus_h = x_values - dist # values for x-h
        x_plus_h = x_values + dist # values for x+h

        f_x_minus = func(x_minus_h) # f(x-h)
        func_values = func(x_values) # f(x)
        f_x_plus = func(x_plus_h) # f(x+h)

        # Calculate the approximations of 1st and 2nd derivation (difference quotient)
        deriv_1_approx = (f_x_plus - func_values) / dist
        deriv_2_approx = (f_x_plus - 2 * func_values + f_x_minus) / dist**2

        return (deriv_1_approx, deriv_2_approx) # return approximation-data



    def calc_approx_of_derivations(self, dist):
        """
        Calculating the approximation of the (1st and 2nd derivative).

        :type dist: float
        :param dist: Value for 'h' in the work description

        Calulating ( f(x+dist) - f(x) ) / h
        and
        (f(x+dist - f*f(x) + f(x-h)) / h^2.

        **Returns: [deriv_1_approx, deriv_2_approx]** list of two arrays (numpy)

        The results will also be stored in the object.
        """

        dist = abs(dist) # absolute value, just in case someone uses negative values

        # calculating the approximations
        deriv_1_approx, deriv_2_approx = self.calc_difference_quotient(self.data['plotting_points'], self.function, dist)

        # save calculated approximation data in the object
        self.data['d1_approx'] = deriv_1_approx
        self.data['d2_approx'] = deriv_2_approx

        return (deriv_1_approx, deriv_2_approx) # return approximation-data



    def calc_errors(self):
        """
        Error calculation of the first and second derivative-approximations.
        This method calculates the maximum of the (absolute) difference between theoretical and approximated value
        of all plotpoints.
        (See task description 1.1, page 3.)

        The approximations of the derivatives (aka difference quotient) must have been calculated before.
        Call ``calc_approx_of_derivations()`` for that purpose just before calling ``calc_errors()``.
        If there is no approximation-data, err-message to stdout and graphics-display
        will be printed, and None will be returned.

        Parameters: none

        Return-values: Error-values will be returned.
        Type: (numpy.float64, numpy.float64)
        """

        message = "Approximation has not been calculated. Can't calculate errors."

        try:
            # calculating the error-maximum
            e_h_1 = max(abs(self.data['ableitung_1_theoretisch'] - self.data['d1_approx']))
            e_h_2 = max(abs(self.data['ableitung_2_theoretisch'] - self.data['d2_approx']))

        except KeyError:
            print(message)
            return None, None

        return e_h_1, e_h_2 # return the error-values


    ############ DRAWING FUNCTIONS ############

    def init_functionplot(self):
        """
        Initializes graphics for the function-plot.
        """
        # Initialising the graphics
        fig = plt.figure()
        ax_f = plt.subplot2grid((3, 1), (0, 0))  # subplot: function
        ax_d1 = plt.subplot2grid((3, 1), (1, 0)) # subplot: 1st derivative
        ax_d2 = plt.subplot2grid((3, 1), (2, 0)) # subplot: 2nd derivative
        fig.subplots_adjust(hspace=0.4) # more hspace for the titles of the subplots needed

        # store graphics-entities as object properties
        self.graphics['fig'] = fig
        self.graphics['ax_f'] = ax_f
        self.graphics['ax_d1'] = ax_d1
        self.graphics['ax_d2'] = ax_d2



    def init_errorplot(self):
        """
        Initializes graphics for the errorvalues-plot.
        """
        # Initialising the graphics
        errfig = plt.figure()
        ax_err = plt.gca()

        # Big title for all three graph-plots f(x), f'(x), f''(x)
        suptitle = str.format("Error Plot (Fehler-Plot)")
        errfig.suptitle(suptitle)

        # store graphics-entities as object properties
        self.graphics['errfig'] = errfig
        self.graphics['ax_err'] = ax_err



    def draw_function(self):
        """
        Plotting the function.

        **Parameter: none**
        """

        ax_f = self.graphics['ax_f']
        ax_f.plot(self.data['plotting_points'], self.data['funktionswerte'], label='f(x)')
        ax_f.set_title('Function Plot (Funktionsplot)')
        ax_f.legend()



    def draw_derivatives_exact(self):
        """
        Plotting the exact derivatives.

        Parameter: none
        """

        # some shorthands
        plotpts = self.data['plotting_points']
        erste_abl = self.data['ableitung_1_theoretisch']
        zweite_abl = self.data['ableitung_2_theoretisch']

        # plot FIRST (THEORETICAL) derivative
        ax_d1 = self.graphics['ax_d1']
        ax_d1.plot(plotpts, erste_abl, label="f'(x)", color='g') # plot deriv_1
        ax_d1.set_title('First Derivative (Erste Ableitung)')

        # set legend for 1st Deriv.
        ax_d1.legend()

        # plot SECOND (THEORETICAL) derivative
        ax_d2 = self.graphics['ax_d2']
        ax_d2.plot(plotpts, zweite_abl, label="f''(x)", color='g') # plot deriv_2
        ax_d2.set_title('Second Derivative (Zweite Ableitung)')

        # set legend for 2nd Deriv.
        ax_d2.legend()



    def draw_derivatives_approximated(self, dist):
        """
        Plotting the approximated derivatives.

        :type dist: float
        :param dist: Value for 'h' in task description

        """

        plotpts = self.data['plotting_points']

        # compute first and second derivative-approximations
        erste_approx, zweite_approx = self.calc_approx_of_derivations(dist)

        # plot FIRST (APPROXIMATED) derivative
        ax_d1 = self.graphics['ax_d1']
        ax_d1.plot(plotpts, erste_approx, "--", label="f'_approx(x)", color='r')

        # set legend for 1st Deriv.
        ax_d1.legend()

        # plot SECOND (APPROXIMATED) derivative
        ax_d2 = self.graphics['ax_d2']
        ax_d2.plot(plotpts, zweite_approx, "--", label="f''_approx(x)", color='r')

        # set legend for 2nd Deriv.
        ax_d2.legend()

        # saving the approximation-data as atribute-data
        self.data['d1_approx'] = erste_approx
        self.data['d2_approx'] = zweite_approx

        # Big title for all three graph-plots f(x), f'(x), f''(x)
        suptitle = str.format("Function and Derivations (Funktion und Ableitungen)\np={}, h={}", self.num_points, dist)
        self.graphics['fig'].suptitle(suptitle)



    def draw_derivatives_exact_and_approximated(self, dist):
        """
        Plotting the exact and approximated derivatives.

        :type dist: float
        :param dist: Value for 'h' in task description

        """
        self.draw_derivatives_exact()
        self.draw_derivatives_approximated(dist)



    def draw_error_plot(self, h_values, append=False, show_plotpoint_distance=True):
        """
        Draw an error plot for the given h-values (dist-values)

        :type h_values: float
        :param h_values: Maximal h-value

        The interval is logarithmic, created with numpy.array-function logspace.
        """

        # setting h-values according the value of the append-option
        # ---------------------------------------------------------
        if append == True:
            self.data['h_values'] = np.append(self.data['h_values'], np.array(h_values))
        else:
            self.data['h_values'] = np.array(h_values)

        h_values = self.data['h_values'] # use the (possibly) updated h-values

        # set up graphics
        errfig = self.graphics['errfig']
        ax_err = self.graphics['ax_err']
        ax_err.clear()

        ax_err.set_xlabel("h")

        # if allowed, plot a vertical line, indicating the plotpoint-distance ('dist' or 'h')
        if show_plotpoint_distance:
            # Calculating distance between plotpoints.
            # If the plot contains h-values below the plotpoint-distance,
            # then draw a vertical line at that h-value, to indicate
            # possible nonsense values.
            dist_between_plotpoints = (self.intervall_end - self.intervall_begin)/self.num_points

            if h_values.size > 0:
                if h_values.min() < dist_between_plotpoints:
                    ax_err.axvline(x=dist_between_plotpoints, color='k', linestyle=':', label="distance between plot-points")

        # create lists for the error-values (they will be appended in a loop)
        err1_values = []
        err2_values = []

        # calculate the error values (approximations (must) have been calculated before)
        for hval in h_values:
            self.calc_approx_of_derivations(hval) # calculate approximations
            err1, err2 = self.calc_errors() # calculating the error-values

            err1_values.append(err1)
            err2_values.append(err2)

        h_quadrat = h_values ** 2
        h_kubik = h_values ** 3

        # h-values
        plt.loglog(h_values, h_values, "r-", markersize=3)
        plt.loglog(h_values, h_quadrat, "g-", markersize=3)
        plt.loglog(h_values, h_kubik, "b-", markersize=3)

        # err-values
        plt.loglog(h_values, err1_values, "m--", markersize=3)
        plt.loglog(h_values, err2_values, "c--", markersize=3)




    def error_plot_finish(self):
        """
        Grid and Legend added to errorplot.

        Parameters: none
        """
        ax_err = self.graphics['ax_err']
        custom_lines = [Line2D([0], [0], color="r", lw=2),
                        Line2D([0], [0], color="g", lw=2),
                        Line2D([0], [0], color="b", lw=2),
                        Line2D([0], [0], linestyle="--", color="m", lw=2),
                        Line2D([0], [0], linestyle="--", color="c", lw=2),
                        Line2D([0], [0], linestyle=":", color="k", lw=2)
                        ]
        ax_err.legend(custom_lines, ['h', 'h^2', 'h^3', 'err1', 'err2', 'Distance between plotpoints'])

        self.graphics['ax_err'].grid()



    def toggle_grid(self):
        """
        Toggling the grid for all function-subplots.

        Parameters: none.

        Must be called after function and approximations have been plotted.
        """

        self.graphics['ax_f'].grid()
        self.graphics['ax_d1'].grid()
        self.graphics['ax_d2'].grid()


######################################################################


def main():
    """
    Main-function that shows the possibilities of this series.
    """
    print("Demo started from function main.")

    # Creating a Approximation objekts regarding PP-I working task 1.2.
    # So, the function wie analyze is g_1(x) = sin(x), Intervall (a,b) = (0, Pi) and we habe
    # p = 1000 plotpoints.
    app = Approximation((0, np.pi), np.sin, np.cos, lambda x: -1 * np.sin(x), 1000)

    # draw function, the theoretical derivatives and the approximations.
    app.init_functionplot()

    app.draw_function()
    app.draw_derivatives_exact_and_approximated(1) # h = 1
    app.toggle_grid()

    # And now the error-plots
    app.init_errorplot()


    if True:
        # regarding task 1.2, it must be possible to add new h-values step by step to the
        # error-plot. That's what is shown/implemented here.

        h_values = np.array([0.001])
        app.draw_error_plot(h_values, append=True)

        h_values = np.array([0.01])
        app.draw_error_plot(h_values, append=True)

        h_values = np.array([0.1])
        app.draw_error_plot(h_values, append=True)

        h_values = np.array([1])
        app.draw_error_plot(h_values, append=True)

    else:
        # Non-appending mode with some values that show interesting behaviour of the error-values over h
        h_values = np.array([1e-8, 1e-5, 0.001, 0.01, 0.1, 1, 2, 3.14])
        app.draw_error_plot(h_values)

    app.error_plot_finish()

    plt.show() # display the graphics

    print("Demo finished.")


######################################################################

if __name__ == '__main__':
    main()
