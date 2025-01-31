from enum import Enum

class FunctionType(Enum):
    """
    Enum class representing different activation functions.
    """
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    SOFTMAX = "softmax"
