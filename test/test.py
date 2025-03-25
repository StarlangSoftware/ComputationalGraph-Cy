import os
import unittest
import random

from typing import List
from Math.Tensor import Tensor
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.ComputationalNode import ComputationalNode
from ComputationalGraph.FunctionType import FunctionType

random.seed(10)


class TestComputationalGraph(unittest.TestCase):
    def create_input_tensor(self, instance: List[str]) -> Tensor:
        """
        Converts an instance (excluding label) into a NumPy Tensor.
        """
        return Tensor([[float(item) for item in instance[:-1]]])

    def test_iris_dataset(self):
        """
        Tests the computational graph on the Iris dataset.
        """
        label_map = {}
        instances = []
        test_set = []
        data_set = []

        with open(os.path.join("../test", "iris.txt"), "r") as file:
            for line in file:
                instance = line.strip().split(",")
                data_set.append(instance)
                if instance[-1] not in label_map:
                    label_map[instance[-1]] = len(label_map)

        random.shuffle(data_set)
        for i, instance in enumerate(data_set):
            if i >= 120:
                test_set.append(instance)
            else:
                instances.append(instance)

        graph = ComputationalGraph()
        input_node = ComputationalNode(learnable=False, operator="*", isBiased=True)

        m1 = Tensor([[random.uniform(-0.01, 0.01) for _c in range(4)] for _r in range(5)])
        w1 = ComputationalNode(value=m1, operator="*")
        a1 = graph.addEdge(first=input_node, second=w1, isBiased=True)
        a1_sigmoid = graph.addEdge(first=a1, second=FunctionType.SIGMOID, isBiased=True)

        m2 = Tensor([[random.uniform(-0.01, 0.01) for _c in range(20)] for _r in range(5)])
        w2 = ComputationalNode(value=m2, operator="*")
        a2 = graph.addEdge(first=a1_sigmoid, second=w2, isBiased=True)
        a2_sigmoid = graph.addEdge(first=a2, second=FunctionType.SIGMOID, isBiased=True)

        m3 = Tensor([[random.uniform(-0.01, 0.01) for _c in range(len(label_map))] for _r in range(21)])
        w3 = ComputationalNode(value=m3, operator="*")
        a3 = graph.addEdge(first=a2_sigmoid, second=w3, isBiased=False)
        graph.addEdge(first=a3, second=FunctionType.SOFTMAX, isBiased=False)

        # Training loop
        epochs = 100
        learning_rate = 0.1
        etaDecrease = 0.99
        class_list = []
        for _ in range(epochs):
            random.shuffle(instances)
            for instance in instances:
                input_node.setValue(self.create_input_tensor(instance))
                graph.forwardCalculation()
                class_list = [label_map[instance[-1]]]
                graph.backpropagation(learning_rate, class_list)

            learning_rate *= etaDecrease

        # Evaluate on test set
        correct = 0
        for instance in test_set:
            input_node.setValue(self.create_input_tensor(instance))
            class_label = graph.predict()[0]
            if class_label == label_map[instance[-1]]:
                correct += 1
        accuracy = correct / len(test_set)
        print("Acc: ", accuracy)
        # self.assertAlmostEqual(accuracy, 1.0, delta=0.001)

    def test_simple_case(self):
        """
        Tests a simple computational graph case.
        """
        # Initialize the computational graph
        graph = ComputationalGraph()

        # Define nodes
        a0 = ComputationalNode(learnable=False, operator="+", isBiased=False)
        a1 = ComputationalNode(learnable=True, operator="+", isBiased=False)
        a2 = graph.addEdge(first=a0, second=a1, isBiased=False)
        output = graph.addEdge(first=a2, second=FunctionType.SOFTMAX, isBiased=False)

        # Assign values
        a0.setValue(Tensor([[random.uniform(0, 100) for _ in range(3)] for _ in range(1)]))
        a1.setValue(Tensor([[random.uniform(0, 100) for _ in range(3)] for _ in range(1)]))

        # Perform forward and backward propagation
        graph.forwardCalculation()
        true_class = [1]
        graph.backpropagation(0.01, true_class)


if __name__ == "__main__":
    unittest.main()
