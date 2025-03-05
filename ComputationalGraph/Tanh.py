import math

from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class Tanh(Function):
    """
    Implements the Sigmoid activation function.
    """

    def calculate(self, tensor):
        """
        Computes the Sigmoid activation for the given tensor.
        :param tensor: NumPy array representing input values.
        :return: Sigmoid-transformed NumPy array.
        """
        result = Tensor([[0 for _r in range(tensor.shape[1]) ] for _c in range(tensor.shape[0])])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                result.set([i, j], math.tanh(tensor.get([i, j])))
        return result

    def derivative(self, tensor):
        """
        Computes the derivative of the Sigmoid function.
        :param tensor: NumPy array representing Sigmoid output.
        :return: Derivative of the Sigmoid function.
        """
        result = Tensor([[0 for _r in range(tensor.shape[1]) ] for _c in range(tensor.shape[0])])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                result.set([i, j], 1 - (tensor.get([i, j]) * tensor.get([i, j])))
        return result
