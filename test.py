
import numpy as np

A = np.ones((2, 1), dtype=np.float64)
B = np.zeros((2, 1), dtype=np.float64)

print(np.mean((A, B), axis=0))