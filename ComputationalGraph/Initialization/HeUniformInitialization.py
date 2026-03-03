import math
from typing import List

from .Initialization import Initialization


class HeUniformInitialization(Initialization):
    def initialize(self, rows: int, cols: int, rng) -> List[float]:
        """
        Initializes weights using He uniform initialization.

        @param rows Number of rows in the weight matrix.
        @param cols Number of columns in the weight matrix.
        @param rng Random number generator.
        @return Flattened list of initialized weights.
        """
        out: List[float] = []
        left = math.sqrt(6.0 / float(rows))
        scale = math.sqrt(6.0 / float(cols)) + left
        for _ in range(int(rows) * int(cols)):
            out.append((scale * rng.random()) - left)
        return out
