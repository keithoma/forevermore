"""
This small module creates the vector b. See protocol for more information.

Fractals are the essence of bread crumbs.

"""

# Copy pasted from block_matrix.py's static method zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

import numpy as np

def rhs(d, n, f):
    """ Computes the right-hand side vector 'b' for a given function 'f'.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem

    Returns
    -------
    np.array or None
        Vector to the right-hand-side f. Returns None if d > 3.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """
    if d < 1 or n < 2:
        raise ValueError("We require d >= 1 and n >= 2!")

    grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]

    if d == 1:
        return (1 / n) ** 2 * np.array([f(x) for x in grid])
    elif d == 2:
        return (1 / n) ** 2 * np.array([f([y, x]) for x in grid for y in grid])
    elif d == 3:
        return (1 / n) ** 2 * np.array([f([z, y, x]) for x in grid for y in grid for z in grid])
    return None
