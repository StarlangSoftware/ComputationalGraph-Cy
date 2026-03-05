from .Function import Function
from Math.Tensor import Tensor


class ReLU(Function):
    """
    Implements the ReLU activation function.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the ReLU activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return ReLU(x).
        """
        out = [max(0.0, float(v)) for v in value.data]
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the ReLU activation function.

        @param value Output of the ReLU(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        out = []
        for i, val in enumerate(value.data):
            out.append(float(backward.data[i]) if float(val) > 0.0 else 0.0)
        return Tensor(out, value.shape)
