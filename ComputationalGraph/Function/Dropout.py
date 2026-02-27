import random

from .Function import Function
from Math.Tensor import Tensor


class Dropout(Function):
    def __init__(self, p: float, rng: random.Random | None = None):
        self.p = float(p)
        self.random = rng if rng is not None else random.Random()
        self.mask: list[float] = []

    def calculate(self, value: Tensor) -> Tensor:
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
        return Tensor([float(backward.data[i]) * self.mask[i] for i in range(len(self.mask))], value.shape)
