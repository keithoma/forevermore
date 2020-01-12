#! /usr/bin/env python3

import numpy as np
from scipy import linalg

def qr(A):
    return linalg.qr(A)

def full_rank(A):
    R = qr(A)
    

def main():
    A = [[-1, 3/2], [1, -1]]
    print(np.array(A))
    print()
    print(qr(A)[0])
    print()
    print(qr(A)[1])

if __name__ == "__main__":
    main()