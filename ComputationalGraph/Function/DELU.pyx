# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp

from .Function import Function
from Math.Tensor import Tensor


class DELU(Function):
    """
    Implements the DELU activation function.
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, xc: float = 1.25643):
        """
        Creates a DELU function with the given parameters.

        @param a Exponential scaling parameter.
        @param b Denominator scaling parameter.
        @param xc Cutoff value after which the function becomes linear.
        """
        self.a = float(a)
        self.b = float(b)
        self.xc = float(xc)

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the DELU activation for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return DELU(x).
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x
        cdef double a = self.a
        cdef double b = self.b
        cdef double xc = self.xc
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            if x > xc:
                out[i] = x
            else:
                out[i] = (exp(a * x) - 1.0) / b
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the DELU activation function.

        @param value Output of the DELU(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double x, bwd
        cdef double a = self.a
        cdef double b = self.b
        cdef double xc = self.xc
        cdef list out = [0.0] * n

        for i in range(n):
            x = float(value.data[i])
            bwd = float(backward.data[i])
            if x > xc:
                out[i] = bwd
            else:
                out[i] = bwd * ((x * b + 1.0) * (a / b))
        return Tensor(out, value.shape)
