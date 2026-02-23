from __future__ import annotations

from typing import Any, Optional

from Math.Tensor import Tensor


class ComputationalNode:
    """
    Java-parity node used by ComputationalGraph.

    Required API:
      - isLearnable(), isBiased()
      - getFunction()
      - getValue()/setValue()
      - getBackward()/setBackward()
    """

    def __init__(
        self,
        learnable: bool = False,
        function: Any = None,
        isBiased: bool = False,
        operator: Optional[str] = None,
        value: Optional[Tensor] = None,
    ):
        self._learnable = bool(learnable)
        self._function = function
        self._is_biased = bool(isBiased)
        self.operator = operator  # optional legacy field; kept for debugging/compat
        self._value: Optional[Tensor] = value
        self._backward: Optional[Tensor] = None

    def __hash__(self) -> int:
        # identity hashing like Java references in HashMap
        return id(self)

    def __repr__(self) -> str:
        vs = None if self._value is None else self._value.shape
        bs = None if self._backward is None else self._backward.shape
        fn = None if self._function is None else self._function.__class__.__name__
        return (
            f"Node(op={self.operator}, learnable={self._learnable}, biased={self._is_biased}, "
            f"value_shape={vs}, backward_shape={bs}, fn={fn})"
        )

    # --- Java-style API ---
    def isLearnable(self) -> bool:
        return self._learnable

    def setLearnable(self, learnable: bool) -> None:
        self._learnable = bool(learnable)

    def isBiased(self) -> bool:
        return self._is_biased

    def setBiased(self, isBiased: bool) -> None:
        self._is_biased = bool(isBiased)

    def getFunction(self) -> Any:
        return self._function

    def setFunction(self, function: Any) -> None:
        self._function = function

    def getValue(self) -> Optional[Tensor]:
        return self._value

    def setValue(self, v: Optional[Tensor]) -> None:
        self._value = v

    def getBackward(self) -> Optional[Tensor]:
        return self._backward

    def setBackward(self, b: Optional[Tensor]) -> None:
        self._backward = b