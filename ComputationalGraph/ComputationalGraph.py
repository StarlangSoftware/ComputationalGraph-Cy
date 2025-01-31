from Math.Matrix import Matrix 

from collections import defaultdict, deque
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

    def addEdge(self, first, second):
        """
        Adds an edge to the computational graph.
        :param first: Parent node.
        :param second: Child node (ComputationalNode) or function type (e.g., 'SIGMOID').
        :param debug: Whether to print debug information.
        :return: The newly created or linked computational node.
        """
        if isinstance(second, FunctionType):  # FunctionType case
            new_node = ComputationalNode(learnable=False, function_type=second)

            if first not in self.node_map.keys():
                self.node_map[first] = []
            self.node_map[first].append(new_node)
            if new_node not in self.reverse_node_map.keys():
                self.reverse_node_map[new_node] = []
            self.reverse_node_map[new_node].append(first)
            return new_node
        
        elif isinstance(second, ComputationalNode):  # ComputationalNode case
            new_node = ComputationalNode(learnable=False, operator=second.getOperator())
            if first not in self.node_map.keys():
                self.node_map[first] = []
            if second not in self.node_map.keys():
                self.node_map[second] = []
            self.node_map[first].append(new_node)
            self.node_map[second].append(new_node)
            if new_node not in self.reverse_node_map.keys():
                self.reverse_node_map[new_node] = []
            self.reverse_node_map[new_node].append(first)
            self.reverse_node_map[new_node].append(second)
            return new_node
        else:
            raise ValueError("Invalid type for second_or_function. Must be a ComputationalNode or FunctionType.")

    def sort(self, node, visited):
        """
        Recursive helper function to perform depth-first search for topological sorting.
        :param node: The current node being processed.
        :param visited: A set of visited nodes.
        :param node_map: A dictionary representing the graph (node -> list of child nodes).
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

    def topologicalSort(self):
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
                while(queue):
                    sorted_list.append(queue.popleft())
        return list(sorted_list) 
    
    def clearRecursive(self, visited, node):
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

    def clear(self):
        """
        Clears the values and gradients of all nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.clearRecursive(visited, node)

    def updateRecursive(self, visited, node):
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

    def updateValues(self):
        """
        Updates the values of all learnable nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.updateRecursive(visited, node)
    
    def calculateDerivative(self, node, child):
        """
        Calculates the derivative of the child node with respect to the parent node.
        :param node: Parent node.
        :param child: Child node.
        :return: The gradient matrix.
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
            return child.getBackward().elementProduct(function.derivative(child.getValue()))

        else:
            right = self.reverse_node_map.get(child)[1]
            if child.getOperator() == '*':
                if left == node:
                    if len(self.node_map.get(child)) == 1 and self.node_map.get(child)[0] not in self.node_map.keys():
                        return child.getBackward().multiply(right.getValue().transpose())
                    return child.getBackward().partial(0, child.getBackward().getRow()-1, 0, child.getBackward().getColumn() - 2).multiply(right.getValue().transpose())
                return left.getValue().transpose().multiply(child.getBackward())

            elif child.getOperator() == '+':
                return child.getBackward().clone()

            elif child.getOperator() == '-':
                if left == node:
                    return child.getBackward().clone()
                else:
                    return child.getBackward().integer_multiply(-1)
        return None

    def calculateRMinusY(self, output, learning_rate, class_label_index):
        """
        Computes the difference between the predicted and actual values (R - Y).
        :param output: The output node of the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: A list of true class labels (index of the correct class for each sample).
        """
        rows, cols = output.getValue().getRow(), output.getValue().getColumn()
        backward = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                if class_label_index[i] == j:
                    backward.setValue(i, j, (1 - output.getValue().getValue(i, j)) * learning_rate)
                else:
                    backward.setValue(i, j, (-output.getValue().getValue(i, j)) * learning_rate)
        output.setBackward(backward)

    def backpropagation(self, learning_rate, class_label_index):
        """
        Performs backpropagation on the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: The true class labels (as a list of integers).
        """
        sorted_nodes = self.topologicalSort()
        output_node = sorted_nodes.pop(0)  
        self.calculateRMinusY(output_node, learning_rate, class_label_index)
        sorted_nodes.pop(0).setBackward(output_node.getBackward().clone())
        while sorted_nodes:
            node = sorted_nodes.pop(0)  
            for child in self.node_map.get(node):
                if node.getBackward() is None:
                    node.setBackward(self.calculateDerivative(node, child))
                else:
                    node.getBackward().add(self.calculateDerivative(node, child))
        self.updateValues()
        self.clear()

    def getBiased(self, first):
        """
        Add a bias term to the node's value by appending a column of ones.
        """
        biased_value = Matrix(first.getValue().getRow(), first.getValue().getColumn() + 1)
        for i in range(first.getValue().getRow()):
            for j in range(first.getValue().getColumn()):
                biased_value.setValue(i, j, first.getValue().getValue(i, j))
            biased_value.setValue(i, first.getValue().getColumn(), 1.0)
        first.setValue(biased_value)

    def predict(self):
        """
        Perform a forward pass and return predicted class indices.
        """
        class_labels = self.forwardCalculation()
        self.clear()
        return class_labels

    def forwardCalculation(self):
        """
        Perform a forward pass through the computational graph.
        Returns:
            A list of predicted class indices.
        """
        sorted_nodes = self.topologicalSort()
        output_node = sorted_nodes[0]

        while len(sorted_nodes) != 1:
            current_node = sorted_nodes.pop()
            for child in self.node_map.get(current_node):
                if child.getValue() is None:
                    if child.getFunctionType() is not None: 
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
                        child.value = function.calculate(current_node.value)
                    else:
                        if (child.getOperator() == '*') and (not current_node.isLearnable()):
                            self.getBiased(current_node)
                        child.setValue(current_node.getValue().clone())
                else:
                    if child.getFunctionType() == None:
                        if child.getOperator() == '*':
                            if not current_node.isLearnable():
                                self.getBiased(current_node)
                            if child.getValue().getColumn() == current_node.getValue().getRow():
                                child.setValue(child.getValue().multiply(current_node.getValue()))
                            else:
                                child.setValue(current_node.getValue().multiply(child.getValue()))
                        elif child.getOperator() == '+':
                            result = child.getValue().clone()
                            result.add(current_node.getValue())
                            child.setValue(result)
                        elif child.operator == '-':
                            result = child.getValue().clone()
                            result.subtract(current_node.getValue())
                            child.setValue(result)
                        else:
                            raise ValueError(f"Unsupported operator: {child.getOperator()}")

        class_label_indices = []
        for i in range(output_node.getValue().getRow()):
            max_val = float('-inf')
            label_index = -1
            for j in range(output_node.getValue().getColumn()):
                if (max_val < output_node.getValue().getValue(i, j)):
                    max_val = output_node.getValue().getValue(i, j)
                    label_index = j
            class_label_indices.append(label_index)

        return class_label_indices