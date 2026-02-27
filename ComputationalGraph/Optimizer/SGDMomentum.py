from Math.Tensor import Tensor

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from .Optimizer import Optimizer


class SGDMomentum(Optimizer):
    def __init__(self, learningRate: float, etaDecrease: float, momentum: float):
        super().__init__(learningRate, etaDecrease)
        self.momentum = float(momentum)
        self.velocityMap: dict[ComputationalNode, list[float]] = {}

    def setGradients(self, node: ComputationalNode) -> None:
        backward = node.getBackward().data
        new_values = [(1.0 - self.momentum) * float(v) for v in backward]
        if node in self.velocityMap:
            old_v = self.velocityMap[node]
            for i in range(len(new_values)):
                new_values[i] = new_values[i] + old_v[i] * self.momentum
        self.velocityMap[node] = list(new_values)
        new_values = [v * self.learningRate for v in new_values]
        node.setBackward(Tensor(new_values, node.getBackward().shape))
