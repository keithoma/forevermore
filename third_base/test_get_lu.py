import numpy as np
from scipy import linalg

mat = np.array([
    [1, 100, 0],
    [0, 2, 0],
    [0, 0, 3]
])

hard = linalg.inv(np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
]))

def get_lu(mat):
    pc, _, _ = linalg.lu(np.transpose(mat))
    pr, _, _ = linalg.lu(np.matmul(linalg.inv(pr), mat))
    return pc, pr

def opti(mat):
    pass

# print(hard)
# print()
# print(np.matmul(hard, mat))
print()

pr, l, u = linalg.lu(mat)


print("The permutation matrix:")
print(pr)
print()
print("Inverted:")
print(linalg.inv(pr))
print("Transposed:")
print(np.transpose(pr))
print()
print()
print(np.matmul(linalg.inv(pr), mat))
print()
print(np.matmul(mat, linalg.inv(pr)))


# 1. transpose give mat, calculate pr
# 2. invert pr,
