import math
from typing import List

from .Initialization import Initialization


class UniformXavierInitialization(Initialization):
    def initialize(self, rows: int, cols: int, rng) -> List[float]:
        limit = math.sqrt(6.0 / float(rows + cols))
        return [((2.0 * rng.random()) - 1.0) * limit for _ in range(int(rows) * int(cols))]
