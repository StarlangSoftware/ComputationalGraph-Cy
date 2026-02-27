import math

from .Function import Function
from Math.Tensor import Tensor


class Tanh(Function):
    def calculate(self, value: Tensor) -> Tensor:
        out = [math.tanh(float(v)) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        out = []
        for i, val in enumerate(value.data):
            out.append((1.0 - float(val) * float(val)) * float(backward.data[i]))
        return Tensor(out, value.shape)
