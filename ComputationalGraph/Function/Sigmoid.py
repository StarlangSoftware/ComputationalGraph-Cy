import math

from .Function import Function
from Math.Tensor import Tensor


class Sigmoid(Function):
    """
    Implements the Sigmoid activation function.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Sigmoid activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return Sigmoid(x).
        """
        out = [1.0 / (1.0 + math.exp(-float(v))) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Sigmoid activation function.

        @param value Output of the Sigmoid(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        out = []
        for i, val in enumerate(value.data):
            out.append(float(backward.data[i]) * float(val) * (1.0 - float(val)))
        return Tensor(out, value.shape)
