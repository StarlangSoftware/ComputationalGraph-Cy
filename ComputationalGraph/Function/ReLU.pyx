# cython: language_level=3, boundscheck=False, wraparound=False

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
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            out[i] = x if x > 0.0 else 0.0
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the ReLU activation function.

        @param value Output of the ReLU(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double val
        cdef list out = [0.0] * n

        for i in range(n):
            val = float(value.data[i])
            out[i] = float(backward.data[i]) if val > 0.0 else 0.0
        return Tensor(out, value.shape)
