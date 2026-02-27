from __future__ import annotations

from Math.Tensor import Tensor
from .Optimizer import Optimizer
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class StochasticGradientDescent(Optimizer):
    def __init__(self, learningRate: float, etaDecrease: float):
        super().__init__(learningRate, etaDecrease)

    def setGradients(self, node: ComputationalNode) -> None:
        # C++: backward *= learningRate  (elementwise)
        b = node.getBackward()
        scaled = [float(x) * self.learningRate for x in b.data]
        node.setBackward(Tensor(scaled, b.shape))