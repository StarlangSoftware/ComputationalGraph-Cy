from .Function import Function
from Math.Tensor import Tensor


class ReLU(Function):
    def calculate(self, value: Tensor) -> Tensor:
        out = [max(0.0, float(v)) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        out = []
        for i, val in enumerate(value.data):
            out.append(float(backward.data[i]) if float(val) > 0.0 else 0.0)
        return Tensor(out, value.shape)
