from Math.Tensor import Tensor

from .Adam import Adam
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


class AdamW(Adam):
    def __init__(
        self,
        learningRate: float,
        etaDecrease: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weightDecay: float,
    ):
        super().__init__(learningRate, etaDecrease, beta1, beta2, epsilon)
        self.weightDecay = float(weightDecay)

    def setGradients(self, node: ComputationalNode) -> None:
        gradients = self.calculate(node)
        values = node.getValue().data
        for i in range(len(gradients)):
            gradients[i] = gradients[i] + self.learningRate * self.weightDecay * float(values[i])
        node.setBackward(Tensor(gradients, node.getBackward().shape))
