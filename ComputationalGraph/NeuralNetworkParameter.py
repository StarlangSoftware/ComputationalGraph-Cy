from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class NeuralNetworkParameter:
    seed: int
    epoch: int
    optimizer: Optional[Any] = None
    initialization: Optional[Any] = None

    def getSeed(self) -> int:
        return int(self.seed)

    def getEpoch(self) -> int:
        return int(self.epoch)

    def getOptimizer(self) -> Optional[Any]:
        return self.optimizer

    def getInitialization(self) -> Optional[Any]:
        return self.initialization