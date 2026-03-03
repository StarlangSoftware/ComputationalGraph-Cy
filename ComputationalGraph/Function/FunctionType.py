from enum import Enum

class FunctionType(Enum):
    """
    Enum class representing different function types supported by the package.
    """
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    SOFTMAX = "softmax"
    DELU = "delu"
    ELU = "elu"
    DROPOUT = "dropout"
