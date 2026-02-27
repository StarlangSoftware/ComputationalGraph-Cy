from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ComputationalGraph.RandomInitialization import RandomInitialization


@dataclass
class NeuralNetworkParameter:
    seed: int
    epoch: int
    optimizer: Optional[Any] = None
    initialization: Optional[Any] = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.initialization is None:
            self.initialization = RandomInitialization()

    def getSeed(self) -> int:
        return int(self.seed)

    def getEpoch(self) -> int:
        return int(self.epoch)

    def getOptimizer(self) -> Optional[Any]:
        return self.optimizer

    def getInitialization(self) -> Optional[Any]:
        return self.initialization

    def getDropout(self) -> float:
        return float(self.dropout)
