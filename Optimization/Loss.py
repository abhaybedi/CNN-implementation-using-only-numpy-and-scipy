import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        epsilon = np.finfo(float).eps
        
        # Element-wise multiplication with the one-hot label tensor extracts the correct class loss
        loss_values = -np.log(prediction_tensor + epsilon) * label_tensor
        
        # Sum over the entire batch
        return np.sum(loss_values)

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        # E_n = - (y / (y_hat + epsilon))
        return -(label_tensor / (self.prediction_tensor + epsilon))