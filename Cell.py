
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from Node import Node
from Edge import Edge

@dataclass(slots=True)
class Cell:
    
    node: list[Node]

    points: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64)) 
    centr: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    edge: list[Edge] = field(default_factory=lambda: [])
    area: float = 0

    neighbour: list[Cell] = field(default_factory=lambda: [])
    neighbour_id: list[int] = field(default_factory=lambda: [])
    div_U: float = 0
    mom_err: np.ndarray = field(default_factory=lambda: np.ones((2, 1), dtype=np.float64))

    p: float = 0
    T: float = 300
    Qh: float = 0
    U: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64)) # just for bc usage (as BC cells are without nodes)

    rho: float = 1000
    k: float = 1 * 1e-2
    gamma: float = 1.4
    R: float = 287.
    cp: float = 1005.

    def __post_init__(self) -> None:
        if len(self.node) != 3: return
        self.points = np.asarray([[self.node[0].x, self.node[0].y, 1],
                       [self.node[1].x, self.node[1].y, 1],
                       [self.node[2].x, self.node[2].y, 1]])
        
        self.area = np.linalg.det(self.points) / 2
        self.centr = np.atleast_2d(self.points.mean(axis=0)[0:2]).T

        for i, j in enumerate([1, 2, 0]):
            self.edge.append(Edge(self.node[i], self.node[j]))
        
