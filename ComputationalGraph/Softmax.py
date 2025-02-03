import math

from ComputationalGraph.Function import Function
from Math.Matrix import Matrix


class Softmax(Function):
    def calculate(self, matrix):
        result = Matrix(matrix.getRow(), matrix.getColumn())
        for i in range(matrix.getRow()):
            _sum = 0
            for k in range(matrix.getColumn()):
                _sum += math.exp(matrix.getValue(i, k))
            for k in range(matrix.getColumn()):
                result.setValue(i, k, math.exp(matrix.getValue(i, k)) / _sum)
        return result

    def derivative(self, matrix):
        # The derivative of softmax is not directly implemented here
        return None
