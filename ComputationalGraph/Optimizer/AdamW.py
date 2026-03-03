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
        """
        Creates an AdamW optimizer.

        @param learningRate Initial learning rate.
        @param etaDecrease Multiplicative decay factor.
        @param beta1 First-moment decay factor.
        @param beta2 Second-moment decay factor.
        @param epsilon Numerical stability constant.
        @param weightDecay Weight decay coefficient.
        @return None.
        """
        super().__init__(learningRate, etaDecrease, beta1, beta2, epsilon)
        self.weightDecay = float(weightDecay)

    def setGradients(self, node: ComputationalNode) -> None:
        """
        Sets AdamW-adjusted gradients for the given node.

        @param node Learnable node whose gradients will be updated.
        @return None.
        """
        gradients = self.calculate(node)
        values = node.getValue().data
        for i in range(len(gradients)):
            gradients[i] = gradients[i] + self.learningRate * self.weightDecay * float(values[i])
        node.setBackward(Tensor(gradients, node.getBackward().shape))
