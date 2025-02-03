from abc import ABC, abstractmethod

from Math.Matrix import Matrix


class Function(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def calculate(self, matrix: Matrix) -> Matrix:
        """
        Computes the function output for the given matrix.
        :param matrix: NumPy array representing input values.
        :return: Transformed NumPy array.
        """
        pass

    @abstractmethod
    def derivative(self, matrix: Matrix) -> Matrix:
        """
        Computes the derivative of the function.
        :param matrix: NumPy array representing function output.
        :return: Derivative of the function.
        """
        pass
