import numpy as np

n = 10

grid = np.linspace(0.0, 1.0, n, endpoint=False)[1:]

a = [(round(x, 3), round(y, 3)) for x in grid for y in grid]

print(a)