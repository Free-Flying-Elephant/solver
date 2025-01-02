

from typing import Any
from numpy.typing import NDArray

import numpy as np


def gauss_seidel_step(A: NDArray, b: NDArray, x: NDArray, err: NDArray) -> None:

    m: int = np.shape(b)[0]

    x_form: NDArray = x.copy()
    for i in range(m):
        sum_Ax: float = sum([A[i, j] * x[j, 0] for j in range(m) if j != i])
        x[i] = (b[i] - sum_Ax) / A[i, i]
        
    err[0] = np.linalg.norm(x - x_form, ord=np.inf)


# A: NDArray = np.asarray([[4, -1, 0, 0],
#               [-1, 4, -1, 0],
#               [0, -1, 4, -1],
#               [0, 0, -1, 3]], dtype=np.float64)

# b: NDArray = np.atleast_2d(np.asarray([15, 10, 10, 10], dtype=np.float64)).T

# x: NDArray = np.zeros(np.shape(b), dtype=np.float64)

# m: int = np.shape(b)[0]

# for iter in range(10):
#     x_form: NDArray = x.copy()
#     for i in range(m):
#         sum_Ax: float = sum([A[i, j] * x[j, 0] for j in range(m) if j != i])
#         x[i] = (b[i] - sum_Ax) / A[i, i]

#     if np.linalg.norm(x - x_form, ord=np.inf) < 1e-6: break







