# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp

from .Function import Function
from Math.Tensor import Tensor


class Sigmoid(Function):
    """
    Implements the Sigmoid activation function.
    """

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the Sigmoid activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return Sigmoid(x).
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            out[i] = 1.0 / (1.0 + exp(-x))
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Sigmoid activation function.

        @param value Output of the Sigmoid(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double val
        cdef list out = [0.0] * n

        for i in range(n):
            val = float(value.data[i])
            out[i] = float(backward.data[i]) * val * (1.0 - val)
        return Tensor(out, value.shape)
