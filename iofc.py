

import os
import json

import numpy as np


def read_mesh(filepath: str) -> dict[str, list[float]]:
    with open(filepath, 'r') as fh:
        dct: dict[str, list[float]] = json.load(fh)
    return dct

def read_idx(filepath: str) -> list[list[int]]:
    with open(filepath, 'r') as fh:
        lst: list[list[int]] = json.load(fh)
    return lst

def read_bc(filepath: str) -> dict[str, list[int]]:
    with open(filepath, 'r') as fh:
        dct: dict[str, list[int]] = json.load(fh)
    return dct

def read_inp_file(filepath: str, var_dct: dict[str, float]) -> int:
    if not os.path.isfile(filepath): return -2
    with open(filepath, "r") as fh:
        lines: list[str] = fh.readlines()
    for key in var_dct.keys():
        var_dct[key] = np.nan
    for line in lines:
        if len(line.split()) < 3: continue
        try: varvalue: float = float(line.split()[-1])
        except TypeError: continue
        varname: str = line.split()[-3]
        if varname in var_dct: var_dct[varname] = varvalue
        else: return -1
    return 0

