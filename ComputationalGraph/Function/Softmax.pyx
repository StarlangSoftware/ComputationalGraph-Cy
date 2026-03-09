# cython: language_level=3, boundscheck=False, wraparound=False

from __future__ import annotations

from libc.math cimport exp
from typing import List, Tuple

from .Function import Function
from Math.Tensor import Tensor


class Softmax(Function):
    """
    Implements the Softmax activation function over the last tensor dimension.
    """

    def calculate(self, x: Tensor) -> Tensor:
        """
        Computes the Softmax activation for the given tensor.

        @param x The tensor whose values are to be transformed.
        @return Softmax(x).
        """
        cdef Py_ssize_t last, rows, r, base, j
        cdef Py_ssize_t rank
        cdef list row, exps
        cdef double m, s, ev

        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))

        rank = len(x.shape)
        last = x.shape[rank - 1]
        out = Tensor([0.0] * _numel(x.shape), x.shape)

        rows = _numel(x.shape) // last
        for r in range(rows):
            base = r * last
            row = x.data[base: base + last]
            m = max(row)
            exps = [exp(float(v) - m) for v in row]
            s = sum(exps)
            for j in range(last):
                ev = exps[j]
                out.data[base + j] = ev / s
        return out

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Softmax activation function.

        @param value Output of the Softmax(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        cdef Py_ssize_t last_dimension_size, i, j, start_index, index, n
        cdef Py_ssize_t rank
        cdef double total
        cdef List[float] values

        rank = len(value.shape)
        last_dimension_size = value.shape[rank - 1]
        values = []
        total = 0.0
        n = len(value.data)

        for i in range(n):
            total += float(value.data[i]) * float(backward.data[i])
            if (i + 1) % last_dimension_size == 0:
                start_index = i // last_dimension_size
                for j in range(last_dimension_size):
                    index = start_index * last_dimension_size + j
                    values.append(float(backward.data[index]) - total)
                total = 0.0

        return value * Tensor(values, value.shape)


cdef Py_ssize_t _numel(shape: Tuple[int, ...]):
    cdef Py_ssize_t n = 1
    cdef int d
    for d in shape:
        n *= d
    return n
