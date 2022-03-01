from dataclasses import dataclass
from typing import Optional

import numpy as np
from openqasm.ast import Expression, IntegerLiteral


@dataclass
class QubitType:
    size: Optional[Expression]


@dataclass
class Qubit:
    state: np.ndarray

    def __init__(self, size: IntegerLiteral = None):
        size = size.value if size is not None else 1
        self.state = np.full((size, 2), np.nan, dtype=complex)

    def reset(self):
        self.state[:] = (1, 0)
