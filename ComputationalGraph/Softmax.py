import math
from ComputationalGraph.Function import Function
from Math.Tensor import Tensor

class Softmax(Function):
    """
    Implements the Softmax activation function.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the Softmax activation for the given tensor.
        """
        result = Tensor([[0 for _ in range(tensor.shape[1])] for _ in range(tensor.shape[0])], tensor.shape)
        for i in range(tensor.shape[0]):
            exp_values = [math.exp(tensor.get((i, k))) for k in range(tensor.shape[1])]
            _sum = sum(exp_values)
            for k in range(tensor.shape[1]):
                result.set((i, k), exp_values[k] / _sum)  # Fixed indexing
        return result

    def derivative(self, tensor: Tensor) -> Tensor:
        """
        Computes the derivative of the Softmax function.
        """
        rows, cols = tensor.shape
        result = Tensor(
            [[[0 for _ in range(cols)] for _ in range(cols)] for _ in range(rows)],
            (rows, cols, cols)
        )  # Full Jacobian matrix

        for i in range(rows):
            for j in range(cols):
                s_i = tensor.get((i, j))
                for k in range(cols):
                    s_k = tensor.get((i, k))
                    result.set((i, j, k), s_i * (1 - s_k) if j == k else -s_i * s_k)  # Fixed Jacobian calculation
        return result
