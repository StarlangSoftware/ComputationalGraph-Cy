from __future__ import annotations

import random
from typing import List

from .Initialization import Initialization


class RandomInitialization(Initialization):
    """
    C++ RandomInitialization: initialize weights in a small uniform range.
    We use U(-0.01, 0.01) (matches typical usage in existing Python tests you had).
    """

    def __init__(self, low: float = -0.01, high: float = 0.01):
        """
        Creates a random uniform initializer.

        @param low Lower bound of the uniform range.
        @param high Upper bound of the uniform range.
        @return None.
        """
        self.low = float(low)
        self.high = float(high)

    def initialize(self, rows: int, cols: int, rng) -> List[float]:
        """
        Initializes weights using a small uniform random range.

        @param rows Number of rows in the weight matrix.
        @param cols Number of columns in the weight matrix.
        @param rng Random number generator.
        @return Flattened list of initialized weights.
        """
        # rng is expected to be a random.Random-like object
        return [rng.uniform(self.low, self.high) for _ in range(int(rows) * int(cols))]
