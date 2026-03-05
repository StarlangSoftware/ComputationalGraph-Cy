from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Set

from Math.Tensor import Tensor
from ComputationalGraph.Node.ComputationalNode import ComputationalNode


def _numel(shape: tuple[int, ...]) -> int:
    """
    Computes the number of elements in a tensor shape.

    @param shape Tensor shape.
    @return Total number of elements.
    """
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Computes row-major strides for the given shape.

    @param shape Tensor shape.
    @return Stride tuple.
    """
    strides: List[int] = []
    prod = 1
    for d in reversed(shape):
        strides.append(prod)
        prod *= int(d)
    return tuple(reversed(strides))


def _unflatten(flat_index: int, strides: tuple[int, ...]) -> List[int]:
    """
    Converts a flat index to a multidimensional index.

    @param flat_index Flat index.
    @param strides Tensor strides.
    @return Multidimensional index.
    """
    idx: List[int] = []
    for s in strides:
        idx.append(flat_index // s)
        flat_index %= s
    return idx


def _ravel_index(idx: List[int], strides: tuple[int, ...]) -> int:
    """
    Converts a multidimensional index to a flat index.

    @param idx Multidimensional index.
    @param strides Tensor strides.
    @return Flat index.
    """
    return sum(i * s for i, s in zip(idx, strides))


class Optimizer(ABC):
    def __init__(self, learningRate: float, etaDecrease: float):
        """
        Creates an optimizer with the given learning rate schedule.

        @param learningRate Initial learning rate.
        @param etaDecrease Multiplicative decay factor.
        @return None.
        """
        self.learningRate = float(learningRate)
        self.etaDecrease = float(etaDecrease)

    def setLearningRate(self) -> None:
        """
        Updates the learning rate by the decay factor.

        @return None.
        """
        # C++: learningRate *= etaDecrease
        self.learningRate *= self.etaDecrease

    @abstractmethod
    def setGradients(self, node: ComputationalNode) -> None:
        """
        Sets the gradients of the given learnable node according to the optimizer rule.

        @param node Learnable node whose gradients will be updated.
        @return None.
        """
        ...

    def broadcast(self, node: ComputationalNode) -> int:
        """
        Checks whether broadcasting reduction is needed for the given node.

        @param node Learnable node.
        @return Broadcast dimension index, or -1 if no reduction is needed.
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
        Reduces the backward tensor to the node value shape along the given axis.

        @param node Learnable node.
        @param axis Broadcast axis.
        @return None.
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
        """
        Recursively updates learnable nodes reachable from the given node.

        @param visited Set of visited nodes.
        @param node Current node.
        @param nodeMap Graph adjacency map.
        @return None.
        """
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
        """
        Updates all learnable node values in the graph.

        @param nodeMap Graph adjacency map.
        @return None.
        """
        visited: Set[ComputationalNode] = set()
        nodes = list(nodeMap.keys())
        for node in nodes:
            if node not in visited:
                self.updateRecursive(visited, node, nodeMap)
