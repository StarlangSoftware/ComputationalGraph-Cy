from __future__ import annotations

from Math.Tensor import Tensor
from .Optimizer import Optimizer
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class StochasticGradientDescent(Optimizer):
    def __init__(self, learningRate: float, etaDecrease: float):
        """
        Creates a stochastic gradient descent optimizer.

        @param learningRate Initial learning rate.
        @param etaDecrease Multiplicative decay factor.
        @return None.
        """
        super().__init__(learningRate, etaDecrease)

    def setGradients(self, node: ComputationalNode) -> None:
        """
        Scales the backward tensor of the given node by the current learning rate.

        @param node Learnable node whose gradients will be updated.
        @return None.
        """
        # C++: backward *= learningRate  (elementwise)
        b = node.getBackward()
        scaled = [float(x) * self.learningRate for x in b.data]
        node.setBackward(Tensor(scaled, b.shape))
