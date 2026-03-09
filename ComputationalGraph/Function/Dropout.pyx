# cython: language_level=3, boundscheck=False, wraparound=False

import random

from .Function import Function
from Math.Tensor import Tensor


class Dropout(Function):
    """
    Implements the Dropout function.
    """

    def __init__(self, p: float, rng: random.Random | None = None):
        """
        Creates a Dropout function with the given dropout probability.

        @param p Dropout probability.
        @param rng Optional random number generator.
        """
        self.p = float(p)
        self.random = rng if rng is not None else random.Random()
        self.mask: list[float] = []

    def calculate(self, value: Tensor) -> Tensor:
        """
        Computes the dropout output for the given tensor.

        @param value The tensor whose values are to be transformed.
        @return Output tensor after dropout masking and scaling.
        """
        cdef Py_ssize_t i, n = len(value.data)
        cdef double multiplier = 1.0 / (1.0 - self.p)
        cdef double r
        cdef list out = [0.0] * n

        self.mask = [0.0] * n
        for i in range(n):
            r = self.random.random()
            if r > self.p:
                self.mask[i] = multiplier
                out[i] = float(value.data[i]) * multiplier
            else:
                self.mask[i] = 0.0
                out[i] = 0.0
        return Tensor(out, value.shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Dropout function.

        @param value Output of the Dropout function.
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t i, n = len(self.mask)
        cdef list out = [0.0] * n

        for i in range(n):
            out[i] = float(backward.data[i]) * self.mask[i]
        return Tensor(out, value.shape)
