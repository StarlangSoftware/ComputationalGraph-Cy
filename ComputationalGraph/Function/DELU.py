import math

from .Function import Function
from Math.Tensor import Tensor


class DELU(Function):
    def __init__(self, a: float = 1.0, b: float = 2.0, xc: float = 1.25643):
        self.a = float(a)
        self.b = float(b)
        self.xc = float(xc)

    def calculate(self, value: Tensor) -> Tensor:
        out = []
        for v in value.data:
            x = float(v)
            if x > self.xc:
                out.append(x)
            else:
                out.append((math.exp(self.a * x) - 1.0) / self.b)
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        out = []
        for i, v in enumerate(value.data):
            x = float(v)
            bwd = float(backward.data[i])
            if x > self.xc:
                out.append(bwd)
            else:
                out.append(bwd * ((x * self.b + 1.0) * (self.a / self.b)))
        return Tensor(out, value.shape)
