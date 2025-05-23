

from typing import Any
from numpy.typing import NDArray

import numpy as np


def cell_inp_var_init(idx: int) -> dict[str, float | int]:
    var_dct: dict[str, float | int] = {"idx" : idx, "p" : 0., "T" : 0., "k" : 0., "Qh" : 0., "u" : 0., "v" : 0.}
    return var_dct


def gauss_seidel_step(A: NDArray, b: NDArray, x: NDArray, err: NDArray) -> None:

    m: int = np.shape(b)[0]

    x_form: NDArray = x.copy()
    for i in range(m):
        sum_Ax: float = sum([A[i, j] * x[j, 0] for j in range(m) if j != i])
        x[i] = (b[i] - sum_Ax) / A[i, i]
        
    err[0] = np.linalg.norm(x - x_form, ord=np.inf)

