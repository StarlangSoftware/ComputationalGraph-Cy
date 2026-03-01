from __future__ import annotations

from Math.Tensor import Tensor

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Function.Softmax import Softmax


class _InvGraph(ComputationalGraph):
    def getClassLabels(self, outputNode):
        t = outputNode.getValue()
        last = t.shape[-1]
        row = t.data[:last]
        return [max(range(len(row)), key=lambda i: row[i])]


def test_forward_sets_output_value_and_shape():
    g = _InvGraph()

    # biased input => [x1, x2, 1]
    inp = MultiplicationNode(False, True)
    w = MultiplicationNode(Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], (3, 2)))

    a = g.addEdge(inp, w, isBiased=False)
    out = g.addEdge(a, Softmax(), isBiased=False)

    inp.setValue(Tensor([1.0, 1.0], (2,)))

    labels = g.forwardCalculation(enableDropout=False)

    assert out.getValue() is not None
    assert out.getValue().shape[-1] == 2
    assert labels in ([0], [1])  # just ensure it returns something valid
