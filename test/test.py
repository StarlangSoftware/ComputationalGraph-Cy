import os
import unittest
import random

from Math.Matrix import Matrix
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.ComputationalNode import ComputationalNode
from ComputationalGraph.FunctionType import FunctionType


random.seed(42)

class TestComputationalGraph(unittest.TestCase):
    def create_input_matrix(self, instance):
        """
        Converts an instance (excluding label) into a NumPy matrix.
        """
        matrix = Matrix(1, len(instance) -1)
        for i in range(len(instance) -1 ):
            matrix.setValue(0, i, float(instance[i]))
        return matrix

    def test_iris_dataset(self):
        """
        Tests the computational graph on the Iris dataset.
        """
        label_map = {}
        instances = []
        test_set = []
        data_set = []

        with open(os.path.join("test", "iris.txt"), "r") as file:
            for line in file:
                instance = line.strip().split(",")
                data_set.append(instance)
                if instance[-1] not in label_map:
                    label_map[instance[-1]] = len(label_map)
        
        random.shuffle(data_set)
        for i, instance in enumerate(data_set):
            if i >= 135:
                test_set.append(instance)
            else:
                instances.append(instance)

        graph = ComputationalGraph()
        input_node = ComputationalNode(learnable=False, operator="*")

        w1 = ComputationalNode(learnable=True, value=Matrix(5, 4, -0.01, 0.01, seed=1), operator="*")
        a1 = graph.addEdge(input_node, w1)
        a1_sigmoid = graph.addEdge(a1, FunctionType.SIGMOID)

        w2 = ComputationalNode(learnable=True, value=Matrix(5, 20, -0.01, 0.01, seed=2), operator="*")
        a2 = graph.addEdge(a1_sigmoid, w2)
        a2_sigmoid = graph.addEdge(a2, FunctionType.SIGMOID)

        w3 = ComputationalNode(learnable=True, value=Matrix(21, len(label_map), -0.01, 0.01, seed=3), operator="*")
        a3 = graph.addEdge(a2_sigmoid, w3)
        graph.addEdge(a3, FunctionType.SOFTMAX)

        # Training loop
        epochs = 200
        learning_rate = 0.001
        class_list = []
        for _ in range(epochs):
            random.shuffle(instances)
            for instance in instances:
                input_node.value = self.create_input_matrix(instance)
                graph.forwardCalculation()
                class_list.append([label_map[instance[-1]]])
                graph.backpropagation(learning_rate, class_list)

        # Evaluate on test set
        correct = 0
        for instance in test_set:
            input_node.setValue(self.create_input_matrix(instance))
            class_label = graph.predict()[0]
            if class_label == label_map[instance[-1]]:
                correct += 1
        accuracy = correct / len(test_set)
        self.assertAlmostEqual(accuracy, 1.0, delta=0.001)

    def test_simple_case(self):
        """
        Tests a simple computational graph case.
        """
        # Initialize the computational graph
        graph = ComputationalGraph()

        # Define nodes
        a0 = ComputationalNode(learnable=False, operator="+")
        a1 = ComputationalNode(learnable=True, operator="+")
        a2 = graph.addEdge(a0, a1)
        output = graph.addEdge(a2, FunctionType.SOFTMAX)

        # Assign values
        a0.value = Matrix(1, 3, 0, 100, seed=1)
        a1.value = Matrix(1, 3, 0, 100, seed=2)

        # Perform forward and backward propagation
        graph.forwardCalculation()
        true_class = [1]
        graph.backpropagation(0.01, true_class)


if __name__ == "__main__":
    unittest.main()
