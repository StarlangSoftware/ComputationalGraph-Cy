import random

from .Function import Function
from Math.Tensor import Tensor


class Dropout(Function):
    """
    Implements the Dropout function.
    """

    def __init__(self, p: float, rng: random.Random | None = None):
        """
        Creates a Dropout function with the given dropout probability.

        @param p Dropout probability.
        @param rng Optional random number generator.
        """
        self.p = float(p)
        self.random = rng if rng is not None else random.Random()
        self.mask: list[float] = []

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the dropout output for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return Output tensor after dropout masking and scaling.
        """
        self.mask.clear()
        multiplier = 1.0 / (1.0 - self.p)
        out = []
        for old_value in value.data:
            r = self.random.random()
            if r > self.p:
                self.mask.append(multiplier)
                out.append(float(old_value) * multiplier)
            else:
                self.mask.append(0.0)
                out.append(0.0)
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Dropout function.

        @param value Output of the Dropout function.
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        return Tensor([float(backward.data[i]) * self.mask[i] for i in range(len(self.mask))], value.shape)
