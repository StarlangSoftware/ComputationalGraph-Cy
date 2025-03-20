import math
from ComputationalGraph.Function import Function
from Math.Tensor import Tensor

class Sigmoid(Function):
    """
    Implements the Sigmoid activation function.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the Sigmoid activation for the given tensor.
        """
        result = Tensor([[0 for _ in range(tensor.shape[1])] for _ in range(tensor.shape[0])], tensor.shape)
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                val = tensor.get((i, j))
                result.set((i, j), 1 / (1 + math.exp(-val)))  # Optimized
        return result

    def derivative(self, tensor: Tensor) -> Tensor:
        """
        Computes the derivative of the Sigmoid function.
        """
        result = Tensor([[0 for _ in range(tensor.shape[1])] for _ in range(tensor.shape[0])], tensor.shape)
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                val = tensor.get((i, j))
                result.set((i, j), val * (1 - val))  # Optimized calculation
        return result
