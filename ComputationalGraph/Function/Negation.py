from .Function import Function
from Math.Tensor import Tensor


class Negation(Function):
    """
    Implements the Negation function.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Negates the values of the given tensor.

        @param value The tensor whose values are to be negated.
        @return The negated tensor.
        """
        return Tensor([-float(v) for v in value.data], value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Negation function.

        @param value Output of the Negation function.
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        return Tensor([-float(v) for v in backward.data], value.shape)
