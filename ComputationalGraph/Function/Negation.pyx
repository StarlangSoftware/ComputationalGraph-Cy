# cython: language_level=3, boundscheck=False, wraparound=False

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
        cdef Py_ssize_t i, n = len(value.data)
        cdef list out = [0.0] * n

        for i in range(n):
            out[i] = -float(value.data[i])
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Negation function.

        @param value Output of the Negation function.
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(backward.data)
        cdef list out = [0.0] * n

        for i in range(n):
            out[i] = -float(backward.data[i])
        return Tensor(out, value.shape)
