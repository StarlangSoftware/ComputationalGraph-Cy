from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.MultiplicationNode import MultiplicationNode
from ComputationalGraph.ComputationalNode import ComputationalNode
from Math.Tensor import Tensor

def test_type_guards_exist_and_work():
    g = ComputationalGraph()

    mul = MultiplicationNode(False, True)
    plain = ComputationalNode()
    assert g._is_multiplication_node(mul) is True
    assert g._is_multiplication_node(plain) is False

    # concat node: we currently use shim/lambda in _new_concatenated_node
    c = g._new_concatenated_node(dimension=0)
    assert g._is_concatenated_node(c) is True