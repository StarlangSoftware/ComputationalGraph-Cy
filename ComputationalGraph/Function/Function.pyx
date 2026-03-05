from abc import ABC, abstractmethod

from Math.Tensor import Tensor


class Function(ABC):
    """
    Defines the interface for activation and transformation functions used by the
    computational graph.
    """

    @abstractmethod
    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes the function output for the given tensor.

        @param tensor Input tensor.
        @return Output tensor after applying the function.
        """
        pass

    @abstractmethod
    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the local gradient of the function and applies it to the incoming
        backward tensor.

        @param value Output of the function.
        @param backward Incoming backward tensor.
        @return Gradient tensor to be propagated to the previous node.
        """
        pass
