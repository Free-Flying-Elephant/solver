

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Node:

    id: int
    x: float
    y: float
    coord: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))

    U: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))

    def __post_init__(self) -> None:
        self.coord = np.asarray([[self.x], [self.y]], dtype=np.float64)
