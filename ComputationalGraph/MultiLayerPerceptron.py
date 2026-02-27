import random

from Math.Tensor import Tensor

from ComputationalGraph.MultiplicationNode import MultiplicationNode
from ComputationalGraph.NeuralNetwork import NeuralNetwork
from ComputationalGraph.Sigmoid import Sigmoid
from ComputationalGraph.Softmax import Softmax


class MultiLayerPerceptron(NeuralNetwork):
    def train(self, trainSet, parameters) -> None:
        optimizer = parameters.getOptimizer()
        input_node = MultiplicationNode(False, True)
        self.inputNodes.append(input_node)

        in_units_biased = 5
        hidden_units = 6
        num_classes = 3
        rng = random.Random(parameters.getSeed())
        initialization = parameters.getInitialization()

        w1 = MultiplicationNode(Tensor(initialization.initialize(in_units_biased, hidden_units, rng), (in_units_biased, hidden_units)))
        a1 = self.addEdge(input_node, w1, isBiased=False)
        a1_sigmoid = self.addEdge(a1, Sigmoid(), isBiased=True)

        w2 = MultiplicationNode(Tensor(initialization.initialize(hidden_units + 1, num_classes, rng), (hidden_units + 1, num_classes)))
        a2 = self.addEdge(a1_sigmoid, w2, isBiased=False)
        self.addEdge(a2, Softmax(), isBiased=False)

        for _ in range(parameters.getEpoch()):
            for instance in trainSet:
                input_node.setValue(self.createInputTensor(instance))
                self.forwardCalculation()
                self.backpropagation(optimizer, [self.getLabelIndex(instance)])
