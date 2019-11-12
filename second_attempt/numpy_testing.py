import numpy as np

def testing():
    M1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ]

    M2 = [
    ['a', 'b', 'c'],
    ['d', 'e', 'f'],
    ['g', 'h', 'i']
    ]

    M3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
    ]

    M4 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
    ]

    BIG0 = [[M4 for _ in range(0, 3)] for __ in range(0, 3)]

    BIG1 = [[M1, M4, M3],
            [M4, M2, M4],
            [M3, M4, M1]]

    BIG2 = [[BIG1, BIG0, BIG0],
            [BIG0, BIG1, BIG0],
            [BIG0, BIG0, BIG1]]

    # ===========================
    # Should look like this:
    #      [[M1 0  I 0 0 0 0 0 0],
    #       [0  M2 0 0 0 0 0 0 0],
    #       [I  0 M1 0 0 0 0 0 0],
    #       [0  0 0 M1 0 I 0 0 0],
    #       ....

    BLOCK = np.bmat([
    [np.bmat(BIG2[0][0]), np.bmat(BIG2[0][1]), np.bmat(BIG2[0][2])],
    [np.bmat(BIG2[1][0]), np.bmat(BIG2[1][1]), np.bmat(BIG2[1][2])],
    [np.bmat(BIG2[2][0]), np.bmat(BIG2[2][1]), np.bmat(BIG2[2][2])]
    ])

    print(BLOCK)

    # print([np.bmat(BIG2[i][j]) for i in range(0, 2) for j in range(0, 2)])

    # print(BIG2[0][0])
    # print(np.bmat(BIG2[0][0]))




testing()
