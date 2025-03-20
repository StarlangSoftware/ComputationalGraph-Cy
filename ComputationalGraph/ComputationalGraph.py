from typing import List, Set, Optional, Union
from collections import defaultdict, deque

from Math.Tensor import Tensor 
from ComputationalGraph.ComputationalNode import ComputationalNode 
from ComputationalGraph.Softmax import Softmax
from ComputationalGraph.Sigmoid import Sigmoid
from ComputationalGraph.FunctionType import FunctionType
from ComputationalGraph.Tanh import Tanh
from ComputationalGraph.ReLU import ReLU


class ComputationalGraph:
    def __init__(self):
        """
        Initializes the computational graph with node maps for forward and reverse connections.
        """
        self.node_map = defaultdict(list)
        self.reverse_node_map = defaultdict(list)

    def addEdge(self, 
                first: ComputationalNode, 
                second: Optional[Union[ComputationalNode, FunctionType]] = None, 
                isBiased: bool = False) -> ComputationalNode:
        """
        Adds an edge to the computational graph.
        :param 
        second_node: Child node (ComputationalNode) .
        second_function: FunctionType (e.g., 'SIGMOID').
        isBiased: Boolean
        :return: The newly created or linked computational node.
        """        
        if isinstance(second, FunctionType):  # Activation function case
            new_node = ComputationalNode(learnable=False, function_type=second, isBiased=isBiased)
        elif isinstance(second, ComputationalNode):  # Computational node case
            new_node = ComputationalNode(learnable=False, operator=second.getOperator(), isBiased=isBiased)
        else:
            raise ValueError("Invalid type for 'second'. Must be a ComputationalNode or FunctionType.")

        # Establish connections in graph structure
        self.node_map[first].append(new_node)
        self.reverse_node_map[new_node].append(first)

        if isinstance(second, ComputationalNode):
            self.node_map[second].append(new_node)
            self.reverse_node_map[new_node].append(second)
            
        return new_node

    def sort(self, node: ComputationalNode, visited: Set[ComputationalNode]) -> List[ComputationalNode]:
        """
        Recursive helper function to perform depth-first search for topological sorting.
        :param node: The current node being processed.
        :param visited: A set of visited nodes.
        :return: A list representing the partial topological order.
        """
        queue = deque()
        visited.add(node)
        if node in self.node_map:
            for child in self.node_map.get(node):
                if child not in visited:
                    queue.extend(self.sort(child, visited))
        queue.append(node)  
        return queue

    def topologicalSort(self) -> List[ComputationalNode]:
        """
        Performs topological sorting on the computational graph.
        :param node_map: A dictionary representing the graph (node -> list of child nodes).
        :return: A list representing the topological order of the nodes.
        """
        sorted_list = deque()
        visited = set()
        for node in self.node_map:
            if node not in visited:
                queue = self.sort(node, visited)
                while (queue):
                    sorted_list.append(queue.popleft())
        return list(sorted_list) 
    
    def clearRecursive(self, visited: Set[ComputationalNode], node: ComputationalNode) -> None:
        """
        Recursive helper function to clear the values and gradients of nodes.
        """
        visited.add(node)
        if node.isLearnable() == False:
            node.setValue(None)  
        node.setBackward(None)

        if node in self.node_map.keys():
            for child in self.node_map.get(node):
                if child not in visited:
                    self.clearRecursive(visited, child)

    def clear(self) -> None:
        """
        Clears the values and gradients of all nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.clearRecursive(visited, node)

    def updateRecursive(self, visited: Set[ComputationalNode], node: ComputationalNode) -> None:
        """
        Recursive helper function to update the values of learnable nodes.
        """
        visited.add(node)
        if node.isLearnable():
            node.updateValue()  

        if node in self.node_map.keys():
            for child in self.node_map.get(node):
                if child not in visited:
                    self.updateRecursive(visited, child)

    def updateValues(self) -> None:
        """
        Updates the values of all learnable nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.updateRecursive(visited, node)
    
    def calculateDerivative(self, node: ComputationalNode, child: ComputationalNode) -> Tensor:
        """
        Calculates the derivative of the child node with respect to the parent node.
        :param node: Parent node.
        :param child: Child node.
        :return: The gradient tensor.
        """
        left = self.reverse_node_map.get(child)[0]
        if len(self.reverse_node_map.get(child)) == 1:
            function = None
            if child.getFunctionType() == FunctionType.SIGMOID:
                function = Sigmoid()
            elif child.getFunctionType() == FunctionType.TANH:
                function = Tanh()
            elif child.getFunctionType() == FunctionType.RELU:
                function = ReLU()
            elif child.getFunctionType() == FunctionType.SOFTMAX:
                function = Softmax()
            else:
                raise ValueError(f"Unsupported function type: {child.getFunctionType()}")
            return child.getBackward() * function.derivative(child.getValue())  # Optimized element-wise multiplication

        else:
            right = self.reverse_node_map.get(child)[1]
            if child.getOperator() == '*':
                if left == node:
                    if child.isBiased() == False:
                        return child.getBackward().dot(right.getValue().transpose())
                    return child.getBackward().partial([0, 0], [child.getBackward().shape[0] ,child.getBackward().shape[1] - 1]).dot(right.getValue().transpose())
                return left.getValue().transpose().dot(child.getBackward())

            elif child.getOperator() == '+':
                return child.getBackward()

            elif child.getOperator() == '-':
                if left == node:
                    return child.getBackward()
                else:
                    result = child.getBackward()
                    for i in range(result.shape[0]):
                        for j in range(result.shape[1]):
                            result.set([i, j], -result.get([i, j]))
                    return result
        return None

    def calculateRMinusY(self, output: ComputationalNode, learning_rate: float, class_label_index: List[int]) -> None:
        """
        Computes the difference between the predicted and actual values (R - Y).
        :param output: The output node of the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: A list of true class labels (index of the correct class for each sample).
        """
        rows, cols = output.getValue().shape[0], output.getValue().shape[1]
        backward = Tensor([[0 for _c in range(cols)] for _r in range(rows)])
        for i in range(rows):
            for j in range(cols):
                if class_label_index[i] == j:
                    backward.set([i, j], (1 - output.getValue().get([i, j])) * learning_rate)
                else:
                    backward.set([i, j], (-output.getValue().get([i, j])) * learning_rate)
        output.setBackward(backward)

    def backpropagation(self, learning_rate: float, class_label_index: List[int]) -> None:
        """
        Performs backpropagation on the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: The true class labels (as a list of integers).
        """
        sorted_nodes = self.topologicalSort()
        output_node = sorted_nodes.pop(0)  
        self.calculateRMinusY(output_node, learning_rate, class_label_index)
        sorted_nodes.pop(0).setBackward(output_node.getBackward())
        while len(sorted_nodes) != 0:
            node = sorted_nodes.pop(0)  
            for child in self.node_map.get(node):
                if node.getBackward() is None:
                    node.setBackward(self.calculateDerivative(node, child))
                else:
                    for i in range(node.getBackward().shape[0]):
                        for j in range(node.getBackward().shape[1]):
                            node.getBackward().set((i, j), node.getBackward().get((i, j)) + self.calculateDerivative(node, child).get((i, j)))

        self.updateValues()
        self.clear()

    def getBiased(self, first: ComputationalNode) -> None:
        """
        Add a bias term to the node's value by appending a column of ones.
        """
        biased_value = Tensor([[0 for _c in range(first.getValue().shape[1] + 1 )] for _r in range(first.getValue().shape[0])])
        for i in range(first.getValue().shape[0]):
            for j in range(first.getValue().shape[1]):
                biased_value.set([i, j], first.getValue().get([i, j]))
            biased_value.set([i, first.getValue().shape[1]], 1.0)
        first.setValue(biased_value)

    def predict(self) -> List[int]:
        """
        Perform a forward pass and return predicted class indices.
        """
        class_labels = self.forwardCalculation()
        self.clear()
        return class_labels

    def forwardCalculation(self) -> List[int]:
        """
        Perform a forward pass through the computational graph.
        Returns:
            A list of predicted class indices.
        """
        sorted_nodes = self.topologicalSort()
        output_node = sorted_nodes[0]

        while len(sorted_nodes) > 1:
            current_node = sorted_nodes.pop()
            for child in self.node_map.get(current_node):
                if child.getValue() == None:
                    if child.getFunctionType() != None: 
                        function = None
                        if child.getFunctionType() == FunctionType.SIGMOID:
                            function = Sigmoid()
                        elif child.getFunctionType() == FunctionType.TANH:
                            function = Tanh()
                        elif child.getFunctionType() == FunctionType.RELU:
                            function = ReLU()
                        elif child.getFunctionType() == FunctionType.SOFTMAX:
                            function = Softmax()
                        else:
                            raise ValueError(f"Unsupported function type: {child.function_type}")
                        child.setValue(function.calculate(current_node.getValue()))
                    else:
                        if current_node.isBiased():
                            self.getBiased(current_node)
                        child.setValue(current_node.getValue())
                else:
                    if child.getFunctionType() == None:
                        if child.getOperator() == '*':
                            if current_node.isBiased():
                                self.getBiased(current_node)
                            if child.getValue().shape[1] == current_node.getValue().shape[0]:
                                child.setValue(child.getValue().dot(current_node.getValue()))
                            else:
                                child.setValue(current_node.getValue().dot(child.getValue()))
                        elif child.getOperator() == '+':
                            result = child.getValue()
                            result.__add__(current_node.getValue())
                            child.setValue(result)
                        elif child.operator == '-':
                            result = child.getValue().clone()
                            result.__sub__(current_node.getValue())
                            child.setValue(result)
                        else:
                            raise ValueError(f"Unsupported operator: {child.getOperator()}")

        class_label_indices = []
        for i in range(output_node.getValue().shape[0]):
            max_val = float('-inf')
            label_index = -1
            for j in range(output_node.getValue().shape[1]):
                if (max_val < output_node.getValue().get([i, j])):
                    max_val = output_node.getValue().get([i, j])
                    label_index = j
            class_label_indices.append(label_index)

        return class_label_indices