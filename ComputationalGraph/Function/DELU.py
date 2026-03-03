import math

from .Function import Function
from Math.Tensor import Tensor


class DELU(Function):
    """
    Implements the DELU activation function.
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, xc: float = 1.25643):
        """
        Creates a DELU function with the given parameters.

        @param a Exponential scaling parameter.
        @param b Denominator scaling parameter.
        @param xc Cutoff value after which the function becomes linear.
        """
        self.a = float(a)
        self.b = float(b)
        self.xc = float(xc)

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the DELU activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return DELU(x).
        """
        out = []
        for v in value.data:
            x = float(v)
            if x > self.xc:
                out.append(x)
            else:
                out.append((math.exp(self.a * x) - 1.0) / self.b)
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the DELU activation function.

        @param value Output of the DELU(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        out = []
        for i, v in enumerate(value.data):
            x = float(v)
            bwd = float(backward.data[i])
            if x > self.xc:
                out.append(bwd)
            else:
                out.append(bwd * ((x * self.b + 1.0) * (self.a / self.b)))
        return Tensor(out, value.shape)
