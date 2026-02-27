from abc import ABC, abstractmethod

from Math.Tensor import Tensor


class Function(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the function output for the given tensor.
        :param tensor: NumPy array representing input values.
        :return: Transformed NumPy array.
        """
        pass

    @abstractmethod
    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes dL/dx using function output and incoming gradient.
        """
        pass
