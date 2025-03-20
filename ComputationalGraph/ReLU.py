from ComputationalGraph.Function import Function
from Math.Tensor import Tensor

class ReLU(Function):
    """
    Implements the ReLU activation function.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the ReLU activation for the given tensor.
        """
        result = Tensor([[0 for _ in range(tensor.shape[1])] for _ in range(tensor.shape[0])], tensor.shape)
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                val = tensor.get((i, j))
                result.set((i, j), max(0, val))  # Optimized calculation
        return result

    def derivative(self, tensor: Tensor) -> Tensor:
        """
        Computes the derivative of the ReLU function.
        """
        result = Tensor([[0 for _ in range(tensor.shape[1])] for _ in range(tensor.shape[0])], tensor.shape)
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                result.set((i, j), 1 if tensor.get((i, j)) > 0 else 0)  # Fixed edge case
        return result
