import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prediction = None

    def forward(self, input_tensor):
        # Shift input by subtracting the max for numerical stability 
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_tensor = np.exp(shifted_input)
        
        # Calculate probabilities
        self.prediction = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.prediction

    def backward(self, error_tensor):
        # E_{n-1} = y_hat * (E_n - sum(E_{n,j} * y_hat_j))
        error_sum = np.sum(error_tensor * self.prediction, axis=1, keepdims=True)
        return self.prediction * (error_tensor - error_sum)