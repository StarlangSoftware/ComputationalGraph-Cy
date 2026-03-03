from __future__ import annotations

from typing import Dict, List

from .ComputationalNode import ComputationalNode


class ConcatenatedNode(ComputationalNode):
    """
    Port of C++ ConcatenatedNode:

      - dimension: int
      - indexMap: maps node identity -> index in concatenation order
    """

    def __init__(self, dimension: int):
        """
        Creates a concatenation node along the given dimension.

        @param dimension Concatenation dimension.
        @return None.
        """
        # C++: ComputationalNode(false, false, nullptr, Tensor({0})) then valueNull=true
        super().__init__(learnable=False, function=None, isBiased=False, operator=None, value=None)
        self._dimension = int(dimension)

        # Identity-based mapping (closest to C++ object key behavior)
        self._index_map: Dict[int, int] = {}
        self._nodes: List[ComputationalNode] = []  # optional but handy for debugging

        # parity: valueNull=true in C++
        self.setValue(None)

    def getDimension(self) -> int:
        """
        Returns the concatenation dimension.

        @return Concatenation dimension.
        """
        return self._dimension

    def addNode(self, node: ComputationalNode) -> None:
        """
        Registers a parent node in concatenation order.

        @param node Node to be added.
        @return None.
        """
        key = id(node)
        if key not in self._index_map:
            self._index_map[key] = len(self._nodes)
            self._nodes.append(node)

    def getIndex(self, node: ComputationalNode) -> int:
        """
        Returns the concatenation index of the given node.

        @param node Parent node.
        @return Index of the node in concatenation order.
        """
        key = id(node)
        if key not in self._index_map:
            raise KeyError("ConcatenatedNode.getIndex called for a node that was not added via addNode()")
        return self._index_map[key]
