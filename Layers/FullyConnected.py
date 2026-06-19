import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gradient_weights = None
        self._optimizer = None
        self.input_tensor = None
        self.trainable = True

        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size)) # +1 because of the bias

    def forward(self, input_tensor):
        ones = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate((input_tensor, ones), axis=1)
        return self.input_tensor @ self.weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.gradient_weights = self.input_tensor.T @ error_tensor

        error_tensor_previous = error_tensor @ self.weights[:-1, :].T # take out the bias row

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights,
                self.gradient_weights
            )

        return error_tensor_previous

    def initialize(self, weights_initializer, bias_initializer):
        # USE self.weights.shape, NOT weights_initializer.shape
        fan_in = self.weights.shape[0] - 1 # input size, -1 for taking out the bias
        fan_out = self.weights.shape[1] # output size

        weights = weights_initializer.initialize((fan_in, fan_out), fan_in, fan_out) # initialize the real weights, without the bias row
        bias = bias_initializer.initialize((1, fan_out), fan_in, fan_out) # one value per output

        self.weights = np.concatenate((weights, bias), axis=0) # append bias in last row
