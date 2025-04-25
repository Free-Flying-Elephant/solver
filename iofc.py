

import json


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
