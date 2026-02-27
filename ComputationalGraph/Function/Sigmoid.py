import math

from .Function import Function
from Math.Tensor import Tensor


class Sigmoid(Function):
    def calculate(self, value: Tensor) -> Tensor:
        out = [1.0 / (1.0 + math.exp(-float(v))) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        out = []
        for i, val in enumerate(value.data):
            out.append(float(backward.data[i]) * float(val) * (1.0 - float(val)))
        return Tensor(out, value.shape)
