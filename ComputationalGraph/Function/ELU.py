import math

from .Function import Function
from Math.Tensor import Tensor


class ELU(Function):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def calculate(self, value: Tensor) -> Tensor:
        out = []
        for v in value.data:
            x = float(v)
            out.append(self.a * (math.exp(x) - 1.0) if x < 0.0 else x)
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        out = []
        for i, v in enumerate(value.data):
            x = float(v)
            b = float(backward.data[i])
            out.append((x + self.a) * b if x < 0.0 else b)
        return Tensor(out, value.shape)
