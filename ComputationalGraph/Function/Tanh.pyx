import math

from .Function import Function
from Math.Tensor import Tensor


class Tanh(Function):
    """
    Implements the Tanh activation function.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Tanh activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return Tanh(x).
        """
        out = [math.tanh(float(v)) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Tanh activation function.

        @param value Output of the Tanh(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        out = []
        for i, val in enumerate(value.data):
            out.append((1.0 - float(val) * float(val)) * float(backward.data[i]))
        return Tensor(out, value.shape)
