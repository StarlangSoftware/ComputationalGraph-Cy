from __future__ import annotations

from typing import Any, Optional

from Math.Tensor import Tensor

from .ComputationalNode import ComputationalNode


class MultiplicationNode(ComputationalNode):
    """
    Java/C++ parity:

    Constructors:
      - MultiplicationNode(learnable: bool, isBiased: bool, isHadamard: bool=False, priorityNode: node|None=None)
      - MultiplicationNode(weightsTensor: Tensor)  -> learnable=True, value=weightsTensor, isBiased=False
    """

    def __init__(
        self,
        learnable: bool | Tensor = False,
        isBiased: bool = False,
        isHadamard: bool = False,
        priorityNode: Optional[Any] = None,
    ):
        """
        Creates a multiplication node or a learnable weights node.

        @param learnable Boolean flag for node construction or a Tensor for the
        weights-node constructor.
        @param isBiased Whether the node output should be biased.
        @param isHadamard Whether multiplication is Hadamard instead of matrix multiplication.
        @param priorityNode Node that determines multiplication order.
        @return None.
        """
        if isinstance(learnable, Tensor):
            # weights node ctor: learnable=True, fixed operator="*"
            super().__init__(learnable=True, function=None, isBiased=False, operator="*", value=learnable)
            self._is_hadamard = False
            self._priority_node = None
        else:
            super().__init__(learnable=bool(learnable), function=None, isBiased=bool(isBiased), operator="*", value=None)
            self._is_hadamard = bool(isHadamard)
            self._priority_node = priorityNode

    def __repr__(self) -> str:
        """
        Returns the debug representation of the multiplication node.

        @return String representation.
        """
        base = super().__repr__()
        return f"{base[:-1]}, hadamard={self._is_hadamard}, priority={'set' if self._priority_node is not None else 'None'})"

    def isHadamard(self) -> bool:
        """
        Returns whether this node performs Hadamard multiplication.

        @return True if Hadamard multiplication is enabled.
        """
        return self._is_hadamard

    def setHadamard(self, isHadamard: bool) -> None:
        """
        Sets whether this node performs Hadamard multiplication.

        @param isHadamard Hadamard flag.
        @return None.
        """
        self._is_hadamard = bool(isHadamard)

    def getPriorityNode(self) -> Optional[Any]:
        """
        Returns the priority node used for multiplication order.

        @return Priority node or None.
        """
        return self._priority_node

    def setPriorityNode(self, node: Optional[Any]) -> None:
        """
        Sets the priority node used for multiplication order.

        @param node Priority node.
        @return None.
        """
        self._priority_node = node
