"""
Copy pasted from block_matrix.py's static method zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
"""
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
    array or None
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
        return [f(x) for x in grid]
    elif d == 2:
        return [f([y, x]) for x in grid for y in grid]
    elif d == 3:
        return [f([z, y, x]) for x in grid for y in grid for z in grid]
    return None



def funa(x):
    return np.sin(x) / x

n = 10

values = np.linspace(1/n, (n-1)/n, n-1)
b = []
for i in values:
    b.append(funa(i))

print(b)
print("----")

print(rhs(1, n, funa))