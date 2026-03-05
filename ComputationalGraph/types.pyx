from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

from Math.Tensor import Tensor


@runtime_checkable
class FunctionLike(Protocol):
    def calculate(self, tensor: Tensor) -> Tensor:
        ...

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        ...


@runtime_checkable
class InitializationLike(Protocol):
    def initialize(self, rows: int, cols: int, rng) -> list[float]:
        ...


@runtime_checkable
class OptimizerLike(Protocol):
    def setLearningRate(self) -> None:
        ...

    def updateValues(self, nodeMap) -> None:
        ...


GraphNode: TypeAlias = "ComputationalNode"

