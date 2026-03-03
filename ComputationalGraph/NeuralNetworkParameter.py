from __future__ import annotations

from dataclasses import dataclass

from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.types import InitializationLike, OptimizerLike


@dataclass
class NeuralNetworkParameter:
    """
    Stores the training parameters of a neural network.
    """

    seed: int
    epoch: int
    optimizer: OptimizerLike | None = None
    initialization: InitializationLike | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """
        Fills in default initialization when it is not supplied.

        @return None.
        """
        if self.initialization is None:
            self.initialization = RandomInitialization()

    def getSeed(self) -> int:
        """
        Returns the random seed.

        @return Seed value.
        """
        return int(self.seed)

    def getEpoch(self) -> int:
        """
        Returns the number of epochs.

        @return Epoch count.
        """
        return int(self.epoch)

    def getOptimizer(self) -> OptimizerLike | None:
        """
        Returns the optimizer instance.

        @return Optimizer object or None.
        """
        return self.optimizer

    def getInitialization(self) -> InitializationLike | None:
        """
        Returns the initialization strategy.

        @return Initialization object.
        """
        return self.initialization

    def getDropout(self) -> float:
        """
        Returns the dropout probability.

        @return Dropout value.
        """
        return float(self.dropout)
