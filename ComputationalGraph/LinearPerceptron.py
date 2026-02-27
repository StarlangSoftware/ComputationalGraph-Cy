import random

from Math.Tensor import Tensor

from ComputationalGraph.MultiplicationNode import MultiplicationNode
from ComputationalGraph.NeuralNetwork import NeuralNetwork
from ComputationalGraph.Softmax import Softmax


class LinearPerceptron(NeuralNetwork):
    def train(self, trainSet, parameters) -> None:
        optimizer = parameters.getOptimizer()
        input_node = MultiplicationNode(False, True)
        self.inputNodes.append(input_node)

        in_units_biased = 5
        num_classes = 3
        rng = random.Random(parameters.getSeed())
        initialization = parameters.getInitialization()
        initial_weights = initialization.initialize(in_units_biased, num_classes, rng)
        w = MultiplicationNode(Tensor(initial_weights, (in_units_biased, num_classes)))

        a = self.addEdge(input_node, w, isBiased=False)
        self.addEdge(a, Softmax(), isBiased=False)

        for _ in range(parameters.getEpoch()):
            for instance in trainSet:
                input_node.setValue(self.createInputTensor(instance))
                self.forwardCalculation()
                self.backpropagation(optimizer, [self.getLabelIndex(instance)])
