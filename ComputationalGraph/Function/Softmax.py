from __future__ import annotations

import math
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
        # Treat 1D as (1, D)
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))

        last = x.shape[-1]
        out = Tensor([0.0] * _numel(x.shape), x.shape)

        # Softmax per row (all dims except last collapsed)
        rows = _numel(x.shape) // last
        for r in range(rows):
            base = r * last
            row = x.data[base : base + last]
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            for j, ev in enumerate(exps):
                out.data[base + j] = ev / s
        return out

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the Softmax activation function.

        @param value Output of the Softmax(x).
        @param backward Backward tensor.
        @return Gradient value of the corresponding node.
        """
        last_dimension_size = value.shape[-1]
        values: List[float] = []
        total = 0.0

        for i, softmax_value in enumerate(value.data):
            total += float(softmax_value) * float(backward.data[i])
            if (i + 1) % last_dimension_size == 0:
                start_index = i // last_dimension_size
                for j in range(last_dimension_size):
                    index = start_index * last_dimension_size + j
                    values.append(float(backward.data[index]) - total)
                total = 0.0

        return value * Tensor(values, value.shape)


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)
