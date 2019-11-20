import math

n = 5

# dimension 1
n_max = (n - 1)

for i in range(1, n_max + 1):
    print("idx^(-1) ({}) = {} / {} = {}".format(i, i, n, i / n))

print("\n ----- ----- ----- \n")

# dimension 2
n_max = (n - 1) ** 2

for i in range(1, n_max + 1):
    sol1 = math.ceil(i / (n - 1))
    sol2 = i - (sol1 - 1) * (n - 1)
    print("idx^(-1) ({0}) = [{1} / {3}, {2} / {3}] = [{4}, {5}]".format(i, sol1, sol2, n, sol1 / n, sol2 / n))

print("\n ----- ----- ----- \n")

# dimension 3
n_max = (n - 1) ** 3

for i in range(1, n_max + 1):
    sol1 = math.ceil(i / (n - 1) ** 2)
    intermediate = i - (sol1 - 1) * (n - 1) ** 2
    sol2 = math.ceil( intermediate / (n - 1) )
    sol3 = i - ((sol1 - 1) * (n - 1) ** 2) - ((sol2 - 1) * (n - 1))

    print("idx^(-1) ({0}) = [{1}, {2}, {3}]".format(i, sol1, sol2, sol3))

print("\n ----- ----- ----- \n")