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
        base = super().__repr__()
        return f"{base[:-1]}, hadamard={self._is_hadamard}, priority={'set' if self._priority_node is not None else 'None'})"

    def isHadamard(self) -> bool:
        return self._is_hadamard

    def setHadamard(self, isHadamard: bool) -> None:
        self._is_hadamard = bool(isHadamard)

    def getPriorityNode(self) -> Optional[Any]:
        return self._priority_node

    def setPriorityNode(self, node: Optional[Any]) -> None:
        self._priority_node = node