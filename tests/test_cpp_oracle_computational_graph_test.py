from Math.Tensor import Tensor

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.StochasticGradientDescent import StochasticGradientDescent
from tests.deep_network import DeepNetwork
from tests.linear_perceptron import LinearPerceptron
from tests.linear_perceptron_single_unit import LinearPerceptronSingleUnit
from tests.multi_layer_perceptron import MultiLayerPerceptron


class _LinearPerceptronSingleUnitGraph(ComputationalGraph):
    def getClassLabels(self, outputNode):
        t = outputNode.getValue()
        last = t.shape[-1]
        row = t.data[:last]
        return [max(range(len(row)), key=lambda i: row[i])]


def test_cpp_oracle_linear_perceptron_single_unit():
    # C++: vector<Tensor> trainSet; Tensor([1,1], {2})
    trainSet = [Tensor([1.0, 1.0], (2,))]

    graph = _LinearPerceptronSingleUnitGraph()
    params = NeuralNetworkParameter(1, 1, None)

    # C++: Optimizer* optimizer = new StochasticGradientDescent(0.1, 0.99);
    optimizer = StochasticGradientDescent(0.1, 0.99)

    # C++: input = new MultiplicationNode(false, true); inputNodes.push_back(input);
    input_node = MultiplicationNode(False, True)
    graph.inputNodes.append(input_node)

    # C++: weightsTensor shape {3,2} effectively (because biased input)
    weights = Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], (3, 2))
    w = MultiplicationNode(weights)

    # C++: a = addEdge(input, w, false); output = addEdge(a, softmax, false);
    a = graph.addEdge(input_node, w, isBiased=False)
    out = graph.addEdge(a, Softmax(), isBiased=False)

    # forward
    input_node.setValue(trainSet[0])
    labels = graph.forwardCalculation(enableDropout=False)
    assert out.getValue() is not None
    assert labels == [0]  # stable with equal logits under our argmax

    # backprop with class {1} like C++
    before = list(w.getValue().data)
    graph.backpropagation(optimizer, [1])
    after = list(w.getValue().data)
    assert before != after
    assert out.getValue() is None  # cleared after backprop like Java

def test_cpp_oracle_linear_perceptron_param_wiring_smoke():
    # Mirrors C++:
    # graph.train(trainSet, NeuralNetworkParameter(1, 10, new SGD(0.1,0.99)))
    # plus initialization used inside train()

    optimizer = StochasticGradientDescent(0.1, 0.99)
    initialization = RandomInitialization()
    params = NeuralNetworkParameter(seed=1, epoch=10, optimizer=optimizer, initialization=initialization)

    assert params.getSeed() == 1
    assert params.getEpoch() == 10
    assert params.getOptimizer() is optimizer
    assert params.getInitialization() is initialization


def test_cpp_oracle_linear_perceptron_single_unit_train_smoke():
    graph = LinearPerceptronSingleUnit()
    train_set = [Tensor([1.0, 1.0], (2,))]
    graph.train(train_set, NeuralNetworkParameter(1, 1, None))
    assert len(graph.inputNodes) == 1


def test_cpp_oracle_linear_perceptron_train_and_test_smoke():
    graph = LinearPerceptron()
    train_set, test_set = [], []
    graph.createIrisDataset(train_set, test_set, seed=1)
    params = NeuralNetworkParameter(
        seed=1,
        epoch=2,
        optimizer=StochasticGradientDescent(0.1, 0.99),
        initialization=RandomInitialization(),
    )
    graph.train(train_set, params)
    acc = graph.test(test_set)
    assert 0.0 <= acc <= 1.0


def test_cpp_oracle_multilayer_perceptron_train_and_test_smoke():
    graph = MultiLayerPerceptron()
    train_set, test_set = [], []
    graph.createIrisDataset(train_set, test_set, seed=1)
    params = NeuralNetworkParameter(
        seed=1,
        epoch=2,
        optimizer=StochasticGradientDescent(0.1, 0.99),
        initialization=RandomInitialization(),
    )
    graph.train(train_set, params)
    acc = graph.test(test_set)
    assert 0.0 <= acc <= 1.0


def test_cpp_oracle_deep_network_train_and_test_smoke():
    graph = DeepNetwork()
    train_set, test_set = [], []
    graph.createIrisDataset(train_set, test_set, seed=1)
    params = NeuralNetworkParameter(
        seed=1,
        epoch=2,
        optimizer=StochasticGradientDescent(0.1, 0.99),
        initialization=RandomInitialization(),
    )
    graph.train(train_set, params)
    acc = graph.test(test_set)
    assert 0.0 <= acc <= 1.0
