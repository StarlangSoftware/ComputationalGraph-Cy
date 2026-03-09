# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp

from .Function import Function
from Math.Tensor import Tensor


class ELU(Function):
    """
    Implements the ELU activation function.
    """

    def __init__(self, a: float = 1.0):
        """
        Creates an ELU function with the given alpha parameter.

        @param a Alpha coefficient used for negative values.
        """
        self.a = float(a)

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the ELU activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return ELU(x).
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x
        cdef double a = self.a
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            out[i] = a * (exp(x) - 1.0) if x < 0.0 else x
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the ELU activation function.

        @param value Output of the ELU(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x, b
        cdef double a = self.a
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            b = float(backward.data[i])
            out[i] = (x + a) * b if x < 0.0 else b
        return Tensor(out, value.shape)
