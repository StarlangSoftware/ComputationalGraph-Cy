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
        """
        Creates a computational node with optional value and function metadata.

        @param learnable Whether the node is learnable.
        @param isBiased Whether the node output should include bias handling.
        @param function Function associated with the node.
        @param value Initial tensor value.
        @param operator Optional operator marker used for compatibility.
        @param nodeType Concrete node type.
        @return None.
        """
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
        """
        Returns an identity-based hash value.

        @return Hash value of the node.
        """
        # Use identity-based hashing (like pointers in C++)
        return id(self)

    # --- C++ API parity ---
    def isBiased(self) -> bool:
        """
        Returns whether the node is biased.

        @return True if the node is biased.
        """
        return self.biased

    def getFunction(self) -> FunctionLike | None:
        """
        Returns the function attached to the node.

        @return Function object or None.
        """
        return self.function

    def getValue(self) -> Optional[Tensor]:
        """
        Returns the current value tensor of the node.

        @return Value tensor or None if it is marked null.
        """
        return None if self.valueNull else self.value

    def setValue(self, v: Optional[Tensor]) -> None:
        """
        Sets the node value tensor.

        @param v New value tensor or None.
        @return None.
        """
        if v is None:
            self.setValueNull()
            return
        self.value = v
        self.valueNull = False

    def updateValue(self) -> None:
        """
        Updates the node value by adding the backward tensor.

        @return None.
        """
        # C++: value = value.add(backward)
        # Note: optimizer scales backward by learningRate in SGD, then updateValue adds it.
        if self.valueNull:
            raise ValueError("updateValue called while valueNull=True")
        if self.backwardNull:
            raise ValueError("updateValue called while backwardNull=True")
        self.value = self.value + self.backward
        self.valueNull = False

    def isLearnable(self) -> bool:
        """
        Returns whether the node is learnable.

        @return True if the node is learnable.
        """
        return self.learnable

    def getBackward(self) -> Optional[Tensor]:
        """
        Returns the backward tensor of the node.

        @return Backward tensor or None if it is marked null.
        """
        return None if self.backwardNull else self.backward

    def setBackward(self, b: Optional[Tensor]) -> None:
        """
        Sets the backward tensor of the node.

        @param b Backward tensor or None.
        @return None.
        """
        if b is None:
            self.setBackwardNull()
            return
        self.backward = b
        self.backwardNull = False

    def isValueNull(self) -> bool:
        """
        Returns whether the value tensor is null.

        @return True if the value tensor is null.
        """
        return self.valueNull

    def setValueNull(self) -> None:
        """
        Marks the value tensor as null.

        @return None.
        """
        self.value = Tensor([0])
        self.valueNull = True

    def isBackwardNull(self) -> bool:
        """
        Returns whether the backward tensor is null.

        @return True if the backward tensor is null.
        """
        return self.backwardNull

    def setBackwardNull(self) -> None:
        """
        Marks the backward tensor as null.

        @return None.
        """
        self.backward = Tensor([0])
        self.backwardNull = True

    def getNodeType(self) -> NodeType:
        """
        Returns the node type.

        @return Node type enum value.
        """
        return self.nodeType
