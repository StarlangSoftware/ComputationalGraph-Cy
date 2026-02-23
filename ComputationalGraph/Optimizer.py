from __future__ import annotations

from typing import Any, Dict, List

from Math.Tensor import Tensor


class Optimizer:
    def updateValues(self, nodeMap: Dict[Any, List[Any]]) -> None:
        raise NotImplementedError


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _tensor_sub_scaled(value: Tensor, grad: Tensor, lr: float) -> Tensor:
    # value - lr * grad
    if value.shape != grad.shape:
        raise ValueError(f"Shape mismatch in optimizer update: {value.shape} vs {grad.shape}")
    out = Tensor([0.0] * _numel(value.shape), value.shape)
    for i in range(len(value.data)):
        out.data[i] = value.data[i] - lr * grad.data[i]
    return out