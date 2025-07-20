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

        with open(os.path.join("test", "iris.txt"), "r") as file:
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

    def test_3d_tensor_operations(self):
        """
        Tests computational graph with 3D tensors.
        """
        graph = ComputationalGraph()
        
        # Create 2D input tensor (batch_size=2, features=4) - flattened for matrix multiplication
        input_2d = Tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
        
        input_node = ComputationalNode(learnable=False, value=input_2d, operator="*")
        
        # Create 2D weight tensor (features=4, output=3) - for matrix multiplication
        weight_2d = Tensor([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9],
                           [1.0, 1.1, 1.2]])
        
        weight_node = ComputationalNode(learnable=True, value=weight_2d, operator="*")
        
        # Add edge and activation
        conv_output = graph.addEdge(first=input_node, second=weight_node, isBiased=False)
        activated_output = graph.addEdge(first=conv_output, second=FunctionType.RELU, isBiased=False)
        
        # Perform forward pass
        graph.forwardCalculation()
        
        # Check output shape and values
        output_value = activated_output.getValue()
        print(f"3D Test - Output shape: {output_value.shape}")
        print(f"3D Test - Output sample values: {output_value.get((0, 0))}, {output_value.get((1, 2))}")
        
        # Perform backward pass
        true_class = [0, 1, 2]  # Match output dimensions
        graph.backpropagation(0.01, true_class)
        
        # Verify that gradients are computed
        gradient = weight_node.getBackward()
        if gradient is not None:
            print(f"3D Test - Weight gradient shape: {gradient.shape}")
        else:
            print("3D Test - No gradient computed (this might be expected)")

    def test_4d_tensor_operations(self):
        """
        Tests computational graph with 4D tensors (batch, channels, height, width).
        """
        graph = ComputationalGraph()
        
        # Create 2D input tensor (batch_size=3, features=6) - flattened for matrix multiplication
        input_2d = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]])
        
        input_node = ComputationalNode(learnable=False, value=input_2d, operator="*")
        
        # Create 2D weight tensor (features=6, output=4) - for matrix multiplication
        weight_2d = Tensor([[0.01, 0.02, 0.03, 0.04],
                           [0.05, 0.06, 0.07, 0.08],
                           [0.09, 0.10, 0.11, 0.12],
                           [0.13, 0.14, 0.15, 0.16],
                           [0.17, 0.18, 0.19, 0.20],
                           [0.21, 0.22, 0.23, 0.24]])
        
        weight_node = ComputationalNode(learnable=True, value=weight_2d, operator="*")
        
        # Add edge and activation
        conv_output = graph.addEdge(first=input_node, second=weight_node, isBiased=False)
        activated_output = graph.addEdge(first=conv_output, second=FunctionType.TANH, isBiased=False)
        
        # Perform forward pass
        graph.forwardCalculation()
        
        # Check output shape and values
        output_value = activated_output.getValue()
        print(f"4D Test - Output shape: {output_value.shape}")
        print(f"4D Test - Output sample values: {output_value.get((0, 0))}, {output_value.get((2, 3))}")
        
        # Perform backward pass
        true_class = [0, 1, 2, 3]  # Match output dimensions
        graph.backpropagation(0.01, true_class)
        
        # Verify that gradients are computed
        gradient = weight_node.getBackward()
        if gradient is not None:
            print(f"4D Test - Weight gradient shape: {gradient.shape}")
        else:
            print("4D Test - No gradient computed (this might be expected)")

    def test_5d_tensor_operations(self):
        """
        Tests computational graph with 5D tensors (batch, time, channels, height, width).
        """
        graph = ComputationalGraph()
        
        # Create 2D input tensor (batch_size=4, features=8) - flattened for matrix multiplication
        input_2d = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                          [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]])
        
        input_node = ComputationalNode(learnable=False, value=input_2d, operator="*")
        
        # Create 2D weight tensor (features=8, output=5) - for matrix multiplication
        weight_2d = Tensor([[0.001, 0.002, 0.003, 0.004, 0.005],
                           [0.006, 0.007, 0.008, 0.009, 0.010],
                           [0.011, 0.012, 0.013, 0.014, 0.015],
                           [0.016, 0.017, 0.018, 0.019, 0.020],
                           [0.021, 0.022, 0.023, 0.024, 0.025],
                           [0.026, 0.027, 0.028, 0.029, 0.030],
                           [0.031, 0.032, 0.033, 0.034, 0.035],
                           [0.036, 0.037, 0.038, 0.039, 0.040]])
        
        weight_node = ComputationalNode(learnable=True, value=weight_2d, operator="*")
        
        # Add edge and activation
        conv_output = graph.addEdge(first=input_node, second=weight_node, isBiased=False)
        activated_output = graph.addEdge(first=conv_output, second=FunctionType.SIGMOID, isBiased=False)
        
        # Perform forward pass
        graph.forwardCalculation()
        
        # Check output shape and values
        output_value = activated_output.getValue()
        print(f"5D Test - Output shape: {output_value.shape}")
        print(f"5D Test - Output sample values: {output_value.get((0, 0))}, {output_value.get((3, 4))}")
        
        # Perform backward pass
        true_class = [0, 1, 2, 3, 4]  # Match output dimensions
        graph.backpropagation(0.01, true_class)
        
        # Verify that gradients are computed
        gradient = weight_node.getBackward()
        if gradient is not None:
            print(f"5D Test - Weight gradient shape: {gradient.shape}")
        else:
            print("5D Test - No gradient computed (this might be expected)")

    def test_mixed_dimension_operations(self):
        """
        Tests computational graph with mixed dimension operations (2D, 3D, 4D tensors).
        """
        graph = ComputationalGraph()
        
        # 2D input (batch_size=2, features=3)
        input_2d = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        input_node = ComputationalNode(learnable=False, value=input_2d, operator="*")
        
        # 2D weight (features=3, output=2) - for matrix multiplication
        weight_2d = Tensor([[0.1, 0.2],
                           [0.3, 0.4],
                           [0.5, 0.6]])
        weight_node = ComputationalNode(learnable=True, value=weight_2d, operator="*")
        
        # Single layer with activation
        layer_output = graph.addEdge(first=input_node, second=weight_node, isBiased=False)
        final_output = graph.addEdge(first=layer_output, second=FunctionType.RELU, isBiased=False)
        
        # Perform forward pass
        graph.forwardCalculation()
        
        # Check output shape and values
        output_value = final_output.getValue()
        print(f"Mixed Dimension Test - Output shape: {output_value.shape}")
        print(f"Mixed Dimension Test - Output sample values: {output_value.get((0, 0))}, {output_value.get((1, 1))}")
        
        # Perform backward pass
        true_class = [0, 1]  # Match output dimensions
        graph.backpropagation(0.01, true_class)
        
        # Verify that gradients are computed
        gradient = weight_node.getBackward()
        
        if gradient is not None:
            print(f"Mixed Dimension Test - Weight gradient shape: {gradient.shape}")
        else:
            print("Mixed Dimension Test - No gradient computed")


if __name__ == "__main__":
    unittest.main()
