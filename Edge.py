

from dataclasses import dataclass, field
import numpy as np

from Node import Node

@dataclass(slots=True)
class Edge:
    
    N1: Node
    N2: Node
    bc_id: int = 0 # no bc
    L: float = 0.
    U: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    C: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    directional: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    normal: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    unit_normal: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))


    def __post_init__(self) -> None:
        self.C = (self.N2.coord + self.N1.coord) / 2
        self.directional = self.N2.coord - self.N1.coord
        self.L = float(np.linalg.norm(self.directional))
        self.normal = np.linalg.matmul(np.asarray([[0, 1],[-1 , 0]]), self.directional)
        self.unit_normal = self.normal / np.linalg.norm(self.normal)


