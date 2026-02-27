import random

from Math.Tensor import Tensor

from ComputationalGraph.MultiplicationNode import MultiplicationNode
from ComputationalGraph.NeuralNetwork import NeuralNetwork
from ComputationalGraph.Sigmoid import Sigmoid
from ComputationalGraph.Softmax import Softmax


class DeepNetwork(NeuralNetwork):
    def train(self, trainSet, parameters) -> None:
        optimizer = parameters.getOptimizer()
        input_node = MultiplicationNode(False, True)
        self.inputNodes.append(input_node)

        in_units_biased = 5
        hidden1 = 6
        hidden2 = 10
        num_classes = 3
        rng = random.Random(parameters.getSeed())
        initialization = parameters.getInitialization()

        w1 = MultiplicationNode(Tensor(initialization.initialize(in_units_biased, hidden1, rng), (in_units_biased, hidden1)))
        a1 = self.addEdge(input_node, w1, isBiased=False)
        a1_sigmoid = self.addEdge(a1, Sigmoid(), isBiased=True)

        w2 = MultiplicationNode(Tensor(initialization.initialize(hidden1 + 1, hidden2, rng), (hidden1 + 1, hidden2)))
        a2 = self.addEdge(a1_sigmoid, w2, isBiased=False)
        a2_sigmoid = self.addEdge(a2, Sigmoid(), isBiased=True)

        w3 = MultiplicationNode(Tensor(initialization.initialize(hidden2 + 1, num_classes, rng), (hidden2 + 1, num_classes)))
        a3 = self.addEdge(a2_sigmoid, w3, isBiased=False)
        self.addEdge(a3, Softmax(), isBiased=False)

        for _ in range(parameters.getEpoch()):
            for instance in trainSet:
                input_node.setValue(self.createInputTensor(instance))
                self.forwardCalculation()
                self.backpropagation(optimizer, [self.getLabelIndex(instance)])
