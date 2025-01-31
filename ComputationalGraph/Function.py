import numpy as np
from abc import ABC, abstractmethod

class Function(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def calculate(self, matrix: np.ndarray) -> np.ndarray:
        """
        Computes the function output for the given matrix.
        :param matrix: NumPy array representing input values.
        :return: Transformed NumPy array.
        """
        pass

    @abstractmethod
    def derivative(self, matrix: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the function.
        :param matrix: NumPy array representing function output.
        :return: Derivative of the function.
        """
        pass
