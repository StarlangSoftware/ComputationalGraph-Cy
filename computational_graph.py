import numpy as np

from collections import defaultdict, deque
from computational_node import ComputationalNode, FunctionType, Matrix, Sigmoid, Tanh, ReLU, Softmax


class ComputationalGraph:
    def __init__(self):
        """
        Initializes the computational graph with node maps for forward and reverse connections.
        """
        self.node_map = defaultdict(list)
        self.reverse_node_map = defaultdict(list)

    def add_edge(self, first, second):
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
            new_node = ComputationalNode(learnable=False, operator=second.operator)
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

    def df_sort(self, node, visited):
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
            for child in self.node_map[node]:
                if child not in visited:
                    queue.extend(self.df_sort(child, visited))
        queue.append(node)  
        return queue

    def topological_sort(self):
        """
        Performs topological sorting on the computational graph.
        :param node_map: A dictionary representing the graph (node -> list of child nodes).
        :return: A list representing the topological order of the nodes.
        """
        sorted_list = deque()
        visited = set()
        for node in self.node_map:
            if node not in visited:
                queue = self.df_sort(node, visited)
                while(queue):
                    sorted_list.append(queue.popleft())
        return list(sorted_list) 
    
    def clear_recursive(self, visited, node):
        """
        Recursive helper function to clear the values and gradients of nodes.
        """
        visited.add(node)
        if not node.is_learnable:
            node.value = None  
        node.backward = None  

        if node in self.node_map.keys():
            for child in self.node_map[node]:
                if child not in visited:
                    self.clear_recursive(visited, child)

    def clear(self):
        """
        Clears the values and gradients of all nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.clear_recursive(visited, node)

    def update_recursive(self, visited, node):
        """
        Recursive helper function to update the values of learnable nodes.
        """
        visited.add(node)
        if node.is_learnable:
            node.update_value()  

        if node in self.node_map.keys():
            for child in self.node_map[node]:
                if child not in visited:
                    self.update_recursive(visited, child)

    def update_values(self):
        """
        Updates the values of all learnable nodes in the graph.
        """
        visited = set()
        for node in self.node_map.keys():
            if node not in visited:
                self.update_recursive(visited, node)
    
    def calculate_derivative(self, node, child):
        """
        Calculates the derivative of the child node with respect to the parent node.
        :param node: Parent node.
        :param child: Child node.
        :return: The gradient matrix.
        """
        left = self.reverse_node_map[child][0]
        if len(self.reverse_node_map[child]) == 1:
            function = None
            if child.function_type == FunctionType.SIGMOID:
                function = Sigmoid()
            elif child.function_type == FunctionType.TANH:
                function = Tanh()
            elif child.function_type == FunctionType.RELU:
                function = ReLU()
            elif child.function_type == FunctionType.SOFTMAX:
                function = Softmax()
            else:
                raise ValueError(f"Unsupported function type: {child.function_type}")
            return child.backward.elementwise_multiply(function.derivative(child.value))

        else:
            right = self.reverse_node_map[child][1]

            if child.operator == '*':
                if left == node:
                    if len(self.node_map[child]) == 1 and self.node_map[child][0] not in self.node_map.keys():
                        return child.backward.multiply(right.value.transpose())
                    return child.backward.partial(0, child.backward.shape[0]-1, 0, child.backward.shape[1] - 2).multiply(right.value.transpose())
                return left.value.transpose().multiply(child.backward)

            elif child.operator == '+':
                return child.backward.clone()

            elif child.operator == '-':
                if left == node:
                    return child.backward.clone()
                else:
                    return child.backward.integer_multiply(-1)
        return None

    def calculate_r_minus_y(self, output, learning_rate, class_label_index):
        """
        Computes the difference between the predicted and actual values (R - Y).
        :param output: The output node of the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: A list of true class labels (index of the correct class for each sample).
        """
        rows, cols = output.value.shape
        backward = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                if class_label_index[i] == j:
                    backward.set_value(i, j, (1 - output.value.get_value(i, j)) * learning_rate)
                else:
                    backward.set_value(i, j, -output.value.get_value(i, j) * learning_rate)
        output.backward = backward    

    def backpropagation(self, learning_rate, class_label_index):
        """
        Performs backpropagation on the computational graph.
        :param learning_rate: The learning rate for gradient descent.
        :param class_label_index: The true class labels (as a list of integers).
        """
        sorted_nodes = self.topological_sort()
        output_node = sorted_nodes.pop(0)  
        self.calculate_r_minus_y(output_node, learning_rate, class_label_index)
        sorted_nodes.pop(0).backward = output_node.backward.clone()
        while sorted_nodes:
            node = sorted_nodes.pop(0)  
            for child in self.node_map[node]:
                if node.backward is None:
                    node.backward = self.calculate_derivative(node, child)
                else:
                    node.backward.add(self.calculate_derivative(node, child))
        self.update_values()
        self.clear()

    def get_biased(self, node):
        """
        Add a bias term to the node's value by appending a column of ones.
        """
        if node.value is None:
            raise ValueError("Node value is None, cannot apply bias.")

        rows, cols = node.value.shape
        biased_value = np.ones((rows, cols + 1))
        biased_value[:, :-1] = node.value.data
        node.value = Matrix(*biased_value.shape).from_array(biased_value)

    def predict(self):
        """
        Perform a forward pass and return predicted class indices.
        """
        class_labels = self.forward_calculation()
        self.clear()
        return class_labels

    def forward_calculation(self):
        """
        Perform a forward pass through the computational graph.
        Returns:
            A list of predicted class indices.
        """
        sorted_nodes = self.topological_sort()
        output_node = sorted_nodes[0]

        while len(sorted_nodes) != 1:
            current_node = sorted_nodes.pop()
            for child in self.node_map[current_node]:
                if child.value is None:
                    if child.function_type is not None: 
                        function = None
                        if child.function_type == FunctionType.SIGMOID:
                            function = Sigmoid()
                        elif child.function_type == FunctionType.TANH:
                            function = Tanh()
                        elif child.function_type == FunctionType.RELU:
                            function = ReLU()
                        elif child.function_type == FunctionType.SOFTMAX:
                            function = Softmax()
                        else:
                            raise ValueError(f"Unsupported function type: {child.function_type}")
                        child.value = function.calculate(current_node.value)
                    else:
                        if (child.operator == '*') and (not current_node.is_learnable):
                            self.get_biased(current_node)
                        child.value = current_node.value.clone()
                else:
                    if child.function_type == None:
                        if child.operator == '*':
                            if not current_node.is_learnable:
                                self.get_biased(current_node)
                            if child.value.shape[1] == current_node.value.shape[0]:
                                child.value = child.value.multiply(current_node.value)
                            else:
                                child.value = current_node.value.multiply(child.value)
                        elif child.operator == '+':
                            result = child.value.clone()
                            result.add(current_node.value)
                            child.value = result.data
                        elif child.operator == '-':
                            result = child.value.clone()
                            result.subtract(current_node.value)
                            child.value = result.data
                        else:
                            raise ValueError(f"Unsupported operator: {child.operator}")

        output_values = output_node.value.data
        class_label_indices = []
        for row in output_values:
            class_label_indices.append(np.argmax(row))

        return class_label_indices