"""
Dieses Programm ist dazu da alle unsere Funktionen f und u für d = 1, 2,3 zu
definieren.

Datum: 02.01.2020
Autoren: hollerju (589259), salihiad (572395)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def list_plotter(values, list_1, list_2, list_3):
    pass
# d = 1
def f_1(vector):
    """Unsere Funktion f für die rechte Seite des Poisson-Problems bzgl. der
    Funktion u aus dem Aufgabenblatt für d = 1.

    Input
    -----
    vector : numpy.ndarray

    Return
    ------
    Langer Return, siehe unten : numpy.float64
    """
    kappa = 10
    return np.pi * kappa * (np.pi*kappa*vector*np.sin(np.pi*kappa*vector)-2*np.cos(np.pi*kappa*vector))

def u_1(vector):
    """ Die Funktion u aus dem Aufgabenblatt 3.2 für d = 1.
    Input
    ------
    vector : numpy.float64

    Return
    ------
    vector * np.sin(kappa * np.pi * vector) : numpy.float64
    """
    kappa = 10
    return vector * np.sin(kappa * np.pi * vector)

# d = 2
def f_2(vector):
    """
    Unsere Funktion f für die rechte Seite des Poisson-Problems bzgl. der Funk-
    tion u aus dem Aufgabenblatt für d = 2.

    Input
    ------
    vector : numpy.ndarray
    Array mit den Inputs x_i als Variablen

    Return
    ------
    Sehr langer Return, siehe unten : numpy.float64
    """
    x_1, x_2 = vector[0], vector[1]
    #Kappa auswählen
    kappa = 10
    return np.pi * kappa * x_2 * np.sin(np.pi*kappa*x_2)*(np.pi * kappa * x_1 *\
np.sin(np.pi * kappa * x_1) - 2*np.cos(np.pi * kappa * x_1)) + np.pi * kappa * \
x_1 * np.sin(np.pi * kappa * x_1) * (np.pi * kappa * x_2 * np.sin(np.pi * kappa\
* x_2) - 2*np.cos(np.pi * kappa * x_2))

def u_2(vector):
    """
    Die Funktion u aus dem Aufgabenblatt 3.2 für d = 2.

    Input:
    ------
    vector : numpy.ndarray
    Unsere Eingabewerte x_i

    Return
    ------
    Langer Return, siehe unten : numpy.float64
    """
    x_1, x_2 = vector[0], vector[1]
    #Kappa auswählen
    kappa = 10
    return x_1*np.sin(np.pi * kappa * x_1) * x_2 * np.sin(np.pi * kappa * x_2)

# d = 3

def f_3(vector):
    """
    Die Funktion f für die rechte Seite des Poisson Problems für unsere Funktion
    u aus dem Aufgabenblatt 3.2 im Fall d = 3.

    Input
    -----
    vector : numpy.ndarray

    Return
    ------
    Sehr langer Return, siehe unten : numpy.float64
    """
    x_1, x_2, x_3 = vector[0], vector[1], vector[2]
    kappa = 10
    return np.pi*kappa*x_2*np.sin(np.pi*kappa*x_2)*x_3*np.sin(np.pi*kappa*x_3)*\
(np.pi*kappa*x_1*np.sin(np.pi*kappa*x_1)-2*np.cos(np.pi*kappa*x_1)) \
+ np.pi*kappa*x_1*np.sin(np.pi*kappa*x_1)*x_3*np.sin(np.pi*kappa*x_3) \
* (np.pi*kappa*x_2-2*np.cos(np.pi*kappa*x_2)) \
+ np.pi*kappa*x_1*np.sin(np.pi*kappa*x_1)*x_2*np.sin(np.pi*kappa*x_2)\
* (np.pi*kappa*x_3 * np.sin(np.pi * kappa * x_3)-2*np.cos(np.pi*kappa*x_3))

def u_3(vector):
    """
    Die Funktion u aus dem Aufgabenblatt 3.2 für d = 3.

    Input
    -----
    vector : numpy.ndarray

    Return
    ------
    Sehr langer Return, siehe unten : numpy.float64
    """
    x_1, x_2, x_3 = vector[0], vector[1], vector[2]
    kappa = 10
    return x_1*np.sin(np.pi*kappa*x_1)*x_2*np.sin(np.pi*kappa*x_2)\
*x_3*np.sin(np.pi*kappa*x_3)
