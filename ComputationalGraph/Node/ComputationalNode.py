from __future__ import annotations

from enum import Enum
from typing import Optional

from Math.Tensor import Tensor
from ComputationalGraph.types import FunctionLike


class NodeType(Enum):
    COMPUTATIONAL_NODE_TYPE = 0
    CONCATENATED_NODE_TYPE = 1
    MULTIPLICATION_NODE_TYPE = 2


class ComputationalNode:
    """
    C++ parity for Node/ComputationalNode.{h,cpp}

    Fields:
      - nodeType
      - value, backward
      - learnable, biased
      - valueNull, backwardNull
      - function

    Key behavior:
      - updateValue(): value = value + backward
      - setValueNull(): value = Tensor({0}), valueNull=True
      - setBackwardNull(): backward = Tensor({0}), backwardNull=True
    """

    def __init__(
        self,
        learnable: bool = False,
        isBiased: bool = False,
        function: FunctionLike | None = None,
        value: Optional[Tensor] = None,
        operator: Optional[str] = None,  # legacy/compat (ignored by core)
        nodeType: NodeType = NodeType.COMPUTATIONAL_NODE_TYPE,
    ):
        self.nodeType: NodeType = nodeType

        self.learnable: bool = bool(learnable)
        self.biased: bool = bool(isBiased)
        self.function: FunctionLike | None = function

        # C++ default Tensor({0}) with null flags set
        self.value: Tensor = value if value is not None else Tensor([0])
        self.backward: Tensor = Tensor([0])

        self.valueNull: bool = value is None
        self.backwardNull: bool = True

        self.operator = operator  # keep for older codepaths

    def __hash__(self) -> int:
        # Use identity-based hashing (like pointers in C++)
        return id(self)

    # --- C++ API parity ---
    def isBiased(self) -> bool:
        return self.biased

    def getFunction(self) -> FunctionLike | None:
        return self.function

    def getValue(self) -> Optional[Tensor]:
        return None if self.valueNull else self.value

    def setValue(self, v: Optional[Tensor]) -> None:
        if v is None:
            self.setValueNull()
            return
        self.value = v
        self.valueNull = False

    def updateValue(self) -> None:
        # C++: value = value.add(backward)
        # Note: optimizer scales backward by learningRate in SGD, then updateValue adds it.
        if self.valueNull:
            raise ValueError("updateValue called while valueNull=True")
        if self.backwardNull:
            raise ValueError("updateValue called while backwardNull=True")
        self.value = self.value + self.backward
        self.valueNull = False

    def isLearnable(self) -> bool:
        return self.learnable

    def getBackward(self) -> Optional[Tensor]:
        return None if self.backwardNull else self.backward

    def setBackward(self, b: Optional[Tensor]) -> None:
        if b is None:
            self.setBackwardNull()
            return
        self.backward = b
        self.backwardNull = False

    def isValueNull(self) -> bool:
        return self.valueNull

    def setValueNull(self) -> None:
        self.value = Tensor([0])
        self.valueNull = True

    def isBackwardNull(self) -> bool:
        return self.backwardNull

    def setBackwardNull(self) -> None:
        self.backward = Tensor([0])
        self.backwardNull = True

    def getNodeType(self) -> NodeType:
        return self.nodeType
