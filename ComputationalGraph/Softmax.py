import math

from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class Softmax(Function):
    def calculate(self, tensor):
        result = Tensor([[0 for _c in range(tensor.shape[0]) ] for _r in range(tensor.shape[1])])
        for i in range(tensor.shape[0]):
            _sum = 0
            for k in range(tensor.shape[1]):
                _sum += math.exp(tensor.get([i, k]))
            for k in range(tensor.shape[1]):
                result.set([i, k], math.exp(tensor.get([i, k])) / _sum)
        return result

    def derivative(self, tensor):
        # The derivative of softmax is not directly implemented here
        return None
