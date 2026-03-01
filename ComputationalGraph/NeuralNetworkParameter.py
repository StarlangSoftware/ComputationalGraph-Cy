from __future__ import annotations

from dataclasses import dataclass

from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.types import InitializationLike, OptimizerLike


@dataclass
class NeuralNetworkParameter:
    seed: int
    epoch: int
    optimizer: OptimizerLike | None = None
    initialization: InitializationLike | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.initialization is None:
            self.initialization = RandomInitialization()

    def getSeed(self) -> int:
        return int(self.seed)

    def getEpoch(self) -> int:
        return int(self.epoch)

    def getOptimizer(self) -> OptimizerLike | None:
        return self.optimizer

    def getInitialization(self) -> InitializationLike | None:
        return self.initialization

    def getDropout(self) -> float:
        return float(self.dropout)
