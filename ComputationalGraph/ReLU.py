from ComputationalGraph.Function import Function
from Math.Matrix import Matrix

class ReLU(Function):
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
                if matrix.getValue(i, j) > 0:
                    result.setValue(i, j, matrix.getValue(i, j))
                else:
                    result.setValue(i, j, 0)
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
                if matrix.getValue(i, j) != 0:
                    result.setValue(i, j, 1)
                else:
                    result.setValue(i, j, 0)
        return result