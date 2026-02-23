from Math.Tensor import Tensor

from ComputationalGraph.ComputationalGraph import ComputationalGraph


class _IdentityFn:
    def calculate(self, x: Tensor) -> Tensor:
        return x

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        return backward


class _ToyGraph(ComputationalGraph):
    def getClassLabels(self, outputNode):
        t = outputNode.getValue()
        last = t.shape[-1]
        row = t.data[:last]
        return [max(range(len(row)), key=lambda i: row[i])]


def test_pytest_collection_smoke():
    g = _ToyGraph()

    inp = g._new_computational_node(learnable=False, function=None, isBiased=False)
    mid = g.addEdge(inp, _IdentityFn(), isBiased=False)
    _out = g.addAdditionEdge(mid, mid, isBiased=False)

    inp.setValue(Tensor([0.1, 0.9], (2,)))
    labels = g.forwardCalculation(enableDropout=False)
    assert labels == [1]