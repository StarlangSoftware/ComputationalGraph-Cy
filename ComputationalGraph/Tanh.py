import numpy as np
import math

from ComputationalGraph.Function import Function
from Math.Matrix import Matrix

class Tanh(Function):
    """
    Implements the Sigmoid activation function.
    """

    def calculate(self, matrix):
        """
        Computes the Sigmoid activation for the given matrix.
        :param matrix: NumPy array representing input values.
        :return: Sigmoid-transformed NumPy array.
        """
        result = Matrix(matrix.getRow(), matrix.getColumn())
        for i in range(matrix.getRow()):
            for j in range(matrix.getColumn()):
                result.setValue(i, j, math.tanh(matrix.getValue(i, j) ))
        return result

    def derivative(self, matrix):
        """
        Computes the derivative of the Sigmoid function.
        :param matrix: NumPy array representing Sigmoid output.
        :return: Derivative of the Sigmoid function.
        """
        result = Matrix(matrix.getRow(), matrix.getColumn())
        for i in range(matrix.getRow()):
            for j in range(matrix.getColumn()):
                result.setValue(i, j, 1 - (matrix.getValue(i, j) * matrix.getValue(i, j)))
        return result
