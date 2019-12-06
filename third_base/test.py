import numpy as np

for n in range(2, 10):
    grid1 = np.linspace(0.0, 1.0, n, endpoint=False)[1:]
    grid2 = np.linspace(1/n, (n-1)/n, n-1)

    print(grid1)
    print(grid2)
