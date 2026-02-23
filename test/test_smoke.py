def test_imports():
    from ComputationalGraph.ComputationalGraph import ComputationalGraph
    from ComputationalGraph.ComputationalNode import ComputationalNode
    from ComputationalGraph.FunctionType import FunctionType
    from Math.Tensor import Tensor

    g = ComputationalGraph()
    a = ComputationalNode(learnable=False, value=Tensor([[1.0, 2.0]]), operator="*")
    assert g is not None
    assert a is not None
