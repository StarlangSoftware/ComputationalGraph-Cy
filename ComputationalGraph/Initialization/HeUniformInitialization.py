import math
from typing import List

from .Initialization import Initialization


class HeUniformInitialization(Initialization):
    def initialize(self, rows: int, cols: int, rng) -> List[float]:
        out: List[float] = []
        left = math.sqrt(6.0 / float(rows))
        scale = math.sqrt(6.0 / float(cols)) + left
        for _ in range(int(rows) * int(cols)):
            out.append((scale * rng.random()) - left)
        return out
