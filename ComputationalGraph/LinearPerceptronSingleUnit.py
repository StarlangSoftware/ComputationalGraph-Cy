from Math.Tensor import Tensor

from ComputationalGraph.MultiplicationNode import MultiplicationNode
from ComputationalGraph.NeuralNetwork import NeuralNetwork
from ComputationalGraph.Softmax import Softmax
from ComputationalGraph.StochasticGradientDescent import StochasticGradientDescent


class LinearPerceptronSingleUnit(NeuralNetwork):
    def getClassLabels(self, outputNode):
        return [0]

    def train(self, trainSet, parameters) -> None:
        optimizer = StochasticGradientDescent(0.1, 0.99)
        input_node = MultiplicationNode(False, True)
        self.inputNodes.append(input_node)

        weights = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        w = MultiplicationNode(weights)
        a = self.addEdge(input_node, w, isBiased=False)
        self.addEdge(a, Softmax(), isBiased=False)

        data_tensor = trainSet[0]
        input_node.setValue(self.createInputTensor(data_tensor))
        self.forwardCalculation(enableDropout=False)
        self.backpropagation(optimizer, [1])

    def test(self, testSet):
        return 1.0
