import numpy as np
from collections import defaultdict, deque
from enum import Enum

class Matrix:
    def __init__(self, rows, cols, low=-0.01, high=0.01, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.data = np.random.uniform(low, high, (rows, cols))
    
    @property
    def shape(self):
        return self.data.shape

    def ndim(self):
        return self.data.ndim

    def reshape(self, new_rows, new_cols):
        reshaped_data = self.data.reshape(new_rows, new_cols)
        return Matrix(new_rows, new_cols).from_array(reshaped_data)
    
    def transpose(self):
        return Matrix(*self.data.T.shape).from_array(self.data.T)
    
    def from_array(self, array):
        self.data = array
        return self
    
    def elementwise_multiply(self, other):
        return Matrix(*self.shape).from_array(self.data * other.data)

    def integer_multiply(self, other):
        return Matrix(*self.shape).from_array(self.data * other)

    def exp(self, other):
        return Matrix(*self.shape).from_array(np.exp(self.data))
    
    def add(self, other):
        self.data += other.data
    
    def subtract(self, other):
        self.data -= other.data
    
    def multiply(self, other):
        return Matrix(*self.data.shape).from_array(np.dot(self.data, other.data))
    
    def clone(self):
        return Matrix(*self.shape).from_array(self.data.copy())
    
    def set_value(self, i, j, value):
        self.data[i, j] = value
    
    def get_value(self, i, j):
        return self.data[i, j]

    def get_all_matrix(self):
        return self.data

    def partial(self, row_start, row_end, col_start, col_end):
        """
        Extract a submatrix based on row and column ranges.
        """
        partial_data = self.data[row_start:row_end + 1, col_start:col_end + 1]
        return Matrix(*partial_data.shape).from_array(partial_data)

class FunctionType(Enum):
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    RELU = "RELU"
    SOFTMAX = "SOFTMAX"

class ActivationFunction:
    """
    Base class for all activation functions.
    """
    def calculate(self, matrix):
        """
        Compute the activation function.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def derivative(self, matrix):
        """
        Compute the derivative of the activation function.
        """
        raise NotImplementedError("Subclasses must implement this method")

class Sigmoid(ActivationFunction):
    def calculate(self, matrix):
        result = (1 / (1 + np.exp(-matrix.data)))
        m = Matrix(rows= 1, cols=result.shape[1])
        return m.from_array(result)

    def derivative(self, matrix):
        sigmoid = self.calculate(matrix)
        delta = sigmoid.data * (1 - sigmoid.data)
        return Matrix(delta.shape[0], delta.shape[1]).from_array(delta)

class Tanh(ActivationFunction):
    def calculate(self, matrix):
        return np.tanh(matrix)

    def derivative(self, matrix):
        return 1 - np.tanh(matrix) ** 2

class ReLU(ActivationFunction):
    def calculate(self, matrix):
        return np.maximum(0, matrix)

    def derivative(self, matrix):
        return np.where(matrix > 0, 1, 0)

class Softmax(ActivationFunction):
    def calculate(self, matrix):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        exp_values = np.exp(matrix.data - np.max(matrix.data, axis=1, keepdims=True))
        exp_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return Matrix(exp_values.shape[0], exp_values.shape[1]).from_array(exp_values)

    def derivative(self, matrix):
        # The derivative of softmax is not directly implemented here
        return None

class ComputationalNode:
    def __init__(self, learnable=False, operator=None, function_type=None, value=None):
        """
        Initializes a ComputationalNode.
        :param learnable: Indicates whether the node is learnable (e.g., weights).
        :param function_type: Type of function (e.g., activation like SIGMOID).
        :param operator: Operator (e.g., '*', '+') for the node.
        :param value: The matrix value associated with the node (optional).
        """
        self.value = value
        self.backward = None
        self.is_learnable = learnable
        self.operator = operator
        self.function_type = function_type

    def __str__(self):
        details = []
        if self.function_type:
            details.append(f"Function: {self.function_type}")
        if self.operator:
            details.append(f"Operator: {self.operator}")
        if self.value:
            details.append(f"Value Shape: {self.value.data.shape}")
        details.append(f"is learnable: {self.is_learnable}")
        return f"Node({', '.join(details)})"

    def __repr__(self):
        return self.__str__()

    def get_function_type(self):
        """
        Returns the function type of the node.
        """
        return self.function_type

    def get_operator(self):
        """
        Returns the operator of the node.
        """
        return self.operator

    def get_value(self):
        """
        Returns the value of the node.
        """
        return self.value

    def set_value(self, value):
        """
        Sets the value of the node.
        :param value: The new value (Matrix object).
        """
        self.value = value

    def update_value(self):
        """
        Update the values.
        """
        if self.value is not None and self.backward is not None:
            for i in range(self.value.shape[0]): 
                for j in range(self.value.shape[1]):
                    self.value.data[i, j] += self.backward.data[i, j]

    def is_learnable(self):
        """
        Returns whether the node is learnable.
        """
        return self.is_learnable

    def get_backward(self):
        """
        Returns the backward gradient of the node.
        """
        return self.backward

    def set_backward(self, backward):
        """
        Sets the backward gradient of the node.
        :param backward: The gradient matrix (Matrix object).
        """
        self.backward = backward


