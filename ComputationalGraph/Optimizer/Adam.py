import math

from Math.Tensor import Tensor

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from .SGDMomentum import SGDMomentum


class Adam(SGDMomentum):
    def __init__(self, learningRate: float, etaDecrease: float, beta1: float, beta2: float, epsilon: float):
        super().__init__(learningRate, etaDecrease, beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.currentBeta1 = 1.0
        self.currentBeta2 = 1.0
        self.momentumMap: dict[ComputationalNode, list[float]] = {}

    def calculate(self, node: ComputationalNode) -> list[float]:
        backward = node.getBackward().data
        new_momentum = [(1.0 - self.momentum) * float(v) for v in backward]
        new_velocity = [(1.0 - self.beta2) * float(v) * float(v) for v in backward]

        if node in self.momentumMap:
            prev_momentum = self.momentumMap[node]
            prev_velocity = self.velocityMap[node]
            for i in range(len(new_momentum)):
                new_momentum[i] += self.momentum * prev_momentum[i]
                new_velocity[i] += self.beta2 * prev_velocity[i]

        self.momentumMap[node] = list(new_momentum)
        self.velocityMap[node] = list(new_velocity)

        corrected_m = [v / (1.0 - self.currentBeta1) for v in new_momentum]
        corrected_v = [v / (1.0 - self.currentBeta2) for v in new_velocity]

        return [
            (corrected_m[i] / (math.sqrt(corrected_v[i]) + self.epsilon)) * self.learningRate
            for i in range(len(corrected_m))
        ]

    def setGradients(self, node: ComputationalNode) -> None:
        gradients = self.calculate(node)
        node.setBackward(Tensor(gradients, node.getBackward().shape))

    def updateValues(self, nodeMap) -> None:
        self.currentBeta1 *= self.momentum
        self.currentBeta2 *= self.beta2
        super().updateValues(nodeMap)
