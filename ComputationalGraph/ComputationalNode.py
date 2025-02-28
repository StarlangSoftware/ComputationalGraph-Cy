from collections import defaultdict, deque
from Math.Matrix import Matrix
from ComputationalGraph.FunctionType import FunctionType

class ComputationalNode:
    def __init__(self, 
                learnable: bool=True, 
                isBiased: bool=False, 
                operator: str=None, 
                function_type: FunctionType=None, 
                value: Matrix=None):
        """
        Initializes a ComputationalNode.
        :param learnable: Indicates whether the node is learnable (e.g., weights).
        :param learnable: Indicates whether the node is biased (e.g., weights).
        :param function_type: Type of function (e.g., activation like SIGMOID).
        :param operator: Operator (e.g., '*', '+') for the node.
        :param value: The matrix value associated with the node (optional).
        """
        self.value = value
        self.backward = None
        self.learnable = learnable
        self.is_biased = isBiased
        self.operator = operator
        self.function_type = function_type

    def __str__(self):
        details = []
        if self.function_type:
            details.append(f"Function: {self.function_type}")
        if self.operator:
            details.append(f"Operator: {self.operator}")
        if self.value:
            details.append(f"Value Shape: ({self.value.getRow()}, {self.value.getColumn()})")
        details.append(f"is learnable: {self.learnable}")
        details.append(f"is biased: {self.is_biased}")
        return f"Node({', '.join(details)})"

    def __repr__(self):
        return self.__str__()

    def isBiased(self):
        """
        Returns whether the node is biased.
        """
        return self.is_biased

    def getFunctionType(self):
        """
        Returns the function type of the node.
        """
        return self.function_type

    def getOperator(self):
        """
        Returns the operator of the node.
        """
        return self.operator

    def getValue(self):
        """
        Returns the value of the node.
        """
        return self.value

    def setValue(self, value):
        """
        Sets the value of the node.
        :param value: The new value (Matrix object).
        """
        self.value = value

    def updateValue(self):
        """
        Update the values.
        """
        if self.value is not None and self.backward is not None:
            for i in range(self.value.getRow()): 
                for j in range(self.value.getColumn()):
                    self.value.setValue(i, j, self.value.getValue(i, j) + self.backward.getValue(i, j))

    def isLearnable(self):
        """
        Returns whether the node is learnable.
        """
        return self.learnable

    def getBackward(self):
        """
        Returns the backward gradient of the node.
        """
        return self.backward

    def setBackward(self, backward):
        """
        Sets the backward gradient of the node.
        :param backward: The gradient matrix (Matrix object).
        """
        self.backward = backward


