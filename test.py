import random
from sklearn.utils import shuffle
import numpy as np

from computational_graph import ComputationalGraph
from computational_node import ComputationalNode, FunctionType, Matrix


def create_input_matrix(instance):
    """
    Converts a single data instance into a Matrix object.
    Assumes the last value is the label and the rest are features.
    """
    values = [float(x) for x in instance[:-1]]
    return Matrix(1, len(values)).from_array(np.array(values).reshape(1, -1))


def load_iris_dataset(filepath):
    """
    Loads the Iris dataset and splits it into training and testing sets.
    Returns:
        - label_map: A dictionary mapping class labels to integers.
        - instances: Training data (list of string arrays).
        - test_set: Testing data (list of string arrays).
    """
    label_map = {}
    data_set = []

    with open(filepath, "r") as file:
        for line in file:
            instance = line.strip().split(",")
            data_set.append(instance)
            if instance[-1] not in label_map:
                label_map[instance[-1]] = len(label_map)

    data_set = shuffle(data_set, random_state=42)
    train_size = int(0.9 * len(data_set))  # 90% for training, 10% for testing
    return label_map, data_set[:train_size], data_set[train_size:]


def test1():
    """
    Implements test1: trains a computational graph on the Iris dataset and evaluates accuracy.
    """
    label_map, instances, test_set = load_iris_dataset("iris.txt")

    # Initialize the computational graph
    graph_1 = ComputationalGraph()
    input_node = ComputationalNode(learnable=False, operator="*")
    # Define weights and layers
    w1 = ComputationalNode(learnable=True, value=Matrix(5, 4, -0.01, 0.01, random_seed=1), operator="*")
    a1 = graph_1.add_edge(input_node, w1)
    a1_sigmoid = graph_1.add_edge(a1, FunctionType.SIGMOID)

    w2 = ComputationalNode(learnable=True, value=Matrix(5, 20, -0.01, 0.01, random_seed=2), operator="*")
    a2 = graph_1.add_edge(a1_sigmoid, w2)
    a2_sigmoid = graph_1.add_edge(a2, FunctionType.SIGMOID)

    w3 = ComputationalNode(learnable=True, value=Matrix(21, len(label_map), -0.01, 0.01, random_seed=3), operator="*")
    a3 = graph_1.add_edge(a2_sigmoid, w3)
    graph_1.add_edge(a3, FunctionType.SOFTMAX)

    # Training loop
    epochs = 10
    learning_rate = 0.01

    for epoch in range(epochs):
        instances = shuffle(instances)
        for instance in instances:
            input_matrix = create_input_matrix(instance)
            input_node.value = input_matrix
            true_label = label_map[instance[-1]]
            graph_1.forward_calculation()
            graph_1.backpropagation(learning_rate, [true_label])

    # Evaluation loop
    correct = 0
    for instance in test_set:
        input_node.set_value(create_input_matrix(instance))
        prediction = graph_1.predict()
        true_label = label_map[instance[-1]]
        if prediction[0] == true_label:
            correct += 1

    accuracy = correct / len(test_set)
    print(f"Test Accuracy: {accuracy:.3f}")


def test2():
    """
    Implements test2: simple graph with addition and softmax operations.
    """
    # Initialize the computational graph
    graph_2 = ComputationalGraph()

    # Define nodes
    a0 = ComputationalNode(learnable=False, operator="+")
    a1 = ComputationalNode(learnable=True, operator="+")
    a2 = graph_2.add_edge(a0, a1)
    output = graph_2.add_edge(a2, FunctionType.SOFTMAX)

    # Assign values
    a0.value = Matrix(1, 3, 0, 100, random_seed=1)
    a1.value = Matrix(1, 3, 0, 100, random_seed=2)

    # Perform forward and backward propagation
    graph_2.forward_calculation()
    true_class = [1]
    graph_2.backpropagation(0.01, true_class)


if __name__ == "__main__":
    # Run test1 and test2
    test1()
    test2()
