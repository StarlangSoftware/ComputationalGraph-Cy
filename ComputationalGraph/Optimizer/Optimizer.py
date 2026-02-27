from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Set

from Math.Tensor import Tensor
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: List[int] = []
    prod = 1
    for d in reversed(shape):
        strides.append(prod)
        prod *= int(d)
    return tuple(reversed(strides))


def _unflatten(flat_index: int, strides: tuple[int, ...]) -> List[int]:
    idx: List[int] = []
    for s in strides:
        idx.append(flat_index // s)
        flat_index %= s
    return idx


def _ravel_index(idx: List[int], strides: tuple[int, ...]) -> int:
    return sum(i * s for i, s in zip(idx, strides))


class Optimizer(ABC):
    def __init__(self, learningRate: float, etaDecrease: float):
        self.learningRate = float(learningRate)
        self.etaDecrease = float(etaDecrease)

    def setLearningRate(self) -> None:
        # C++: learningRate *= etaDecrease
        self.learningRate *= self.etaDecrease

    @abstractmethod
    def setGradients(self, node: ComputationalNode) -> None:
        ...

    def broadcast(self, node: ComputationalNode) -> int:
        """
        C++ parity:
        Return the index of the only dimension where:
          value_shape[i] != backward_shape[i] and value_shape[i] == 1
        Else return -1 (no broadcast or ambiguous).
        """
        v = node.getValue().shape
        b = node.getBackward().shape
        index = -1
        for i in range(len(v)):
            if v[i] != b[i]:
                if v[i] == 1:
                    if index != -1:
                        return -1
                    index = i
        return index

    def _reduce_backward_to_value_shape(self, node: ComputationalNode, axis: int) -> None:
        """
        Safer Python implementation of the C++ broadcast reduction:
        sums backward over the broadcasted axis into shape(value).
        """
        v_shape = tuple(node.getValue().shape)
        b_tensor: Tensor = node.getBackward()
        b_shape = tuple(b_tensor.shape)

        v_strides = _strides(v_shape)
        b_strides = _strides(b_shape)

        accum = [0.0] * _numel(v_shape)

        for flat in range(_numel(b_shape)):
            b_idx = _unflatten(flat, b_strides)
            v_idx = list(b_idx)
            v_idx[axis] = 0  # broadcast axis collapsed
            v_flat = _ravel_index(v_idx, v_strides)
            accum[v_flat] += b_tensor.get(tuple(b_idx))

        node.setBackward(Tensor(accum, v_shape))

    def updateRecursive(
        self,
        visited: Set[ComputationalNode],
        node: ComputationalNode,
        nodeMap: Dict[ComputationalNode, List[ComputationalNode]],
    ) -> None:
        visited.add(node)

        if node.isLearnable():
            axis = self.broadcast(node)
            if axis != -1:
                self._reduce_backward_to_value_shape(node, axis)

            self.setGradients(node)
            node.updateValue()

        if node in nodeMap:
            for child in nodeMap[node]:
                if child not in visited:
                    self.updateRecursive(visited, child, nodeMap)

    def updateValues(self, nodeMap: Dict[ComputationalNode, List[ComputationalNode]]) -> None:
        visited: Set[ComputationalNode] = set()
        nodes = list(nodeMap.keys())
        for node in nodes:
            if node not in visited:
                self.updateRecursive(visited, node, nodeMap)