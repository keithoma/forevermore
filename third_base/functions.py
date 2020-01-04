import numpy as np

def u(v, k=1.0):
    """ Example function to draw an example graph.

    Parameters:
    v : list
        The vector containing the grid points. Must not be np.array.
    k : float
        A constant.

    Returns:
    float
        The solution to the equation sum_{l = 1}^d x_l * sin(k * pi * x_l)
    """
    _ = 1
    vector_size = len(v) if isinstance(v, list) else 1
    for l in range(vector_size):
        _ = _ * v[l] * np.sin(k * np.pi * v[l])
    return _

def f(v, k=1.0):
    if len(v) == 1:
        return k * np.pi * (k * np.pi * v[0] * np.sin(k * np. pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))

    elif len(v) == 2:
        sum1 = k * np.pi * v[1] * np.sin(k * np.pi * v[1]) * (k * np.pi * v[0] * np.sin(k * np.pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))
        sum2 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * (k * np.pi * v[1] * np.sin(k * np.pi * v[1]) - 2 * np.cos(k * np.pi * v[1]))
        return sum1 + sum2

    elif len(v) == 3:
        sum1 = k * np.pi * v[1] * np.sin(k * np.pi * v[1]) * k * np.pi * v[2] * np.sin(k * np.pi * v[2])
        sum1 = sum1 * (k * np.pi * v[0] * np.sin(k * np.pi * v[0]) - 2 * np.cos(k * np.pi * v[0]))

        sum2 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * k * np.pi * v[2] * np.sin(k * np.pi * v[2])
        sum2 = sum1 * (k * np.pi * v[1] * np.sin(k * np.pi * v[1]) - 2 * np.cos(k * np.pi * v[1]))

        sum3 = k * np.pi * v[0] * np.sin(k * np.pi * v[0]) * k * np.pi * v[1] * np.sin(k * np.pi * v[1])
        sum3 = sum1 * (k * np.pi * v[2] * np.sin(k * np.pi * v[2]) - 2 * np.cos(k * np.pi * v[2]))
        return sum1 + sum2 + sum3
