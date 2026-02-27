from __future__ import annotations

import math
from typing import List, Tuple

from .Function import Function
from Math.Tensor import Tensor


class Softmax(Function):
    """
    Row-wise softmax over the last dimension (Java parity).
    """

    def calculate(self, x: Tensor) -> Tensor:
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
        Java typically composes Softmax with cross-entropy, where dL/dz = (R - Y).
        In this codebase, backprop starts with R-Y already, so Softmax derivative is identity passthrough.
        """
        return backward


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)
