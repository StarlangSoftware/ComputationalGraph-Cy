from .Function import Function
from Math.Tensor import Tensor


class Negation(Function):
    def calculate(self, value: Tensor) -> Tensor:
        return Tensor([-float(v) for v in value.data], value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        return Tensor([-float(v) for v in backward.data], value.shape)
