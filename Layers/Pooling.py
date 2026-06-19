import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        
        # Cache to store the locations of the maximums for routing the gradient back
        self.max_indices = []

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        b, c, y, x = input_tensor.shape
        
        s_y, s_x = self.stride_shape
        p_y, p_x = self.pooling_shape
        
        # Valid padding logic: border elements that don't fit are discarded
        out_y = int(np.floor((y - p_y) / s_y)) + 1
        out_x = int(np.floor((x - p_x) / s_x)) + 1
        
        output_tensor = np.zeros((b, c, out_y, out_x))
        self.max_indices = []
        
        for b_idx in range(b):
            for c_idx in range(c):
                for i in range(out_y):
                    for j in range(out_x):
                        start_y = i * s_y
                        start_x = j * s_x
                        
                        # Extract pooling kernel boundary
                        region = input_tensor[b_idx, c_idx, start_y:start_y+p_y, start_x:start_x+p_x]
                        
                        output_tensor[b_idx, c_idx, i, j] = np.max(region)
                        
                        # Find exactly where that max came from
                        flat_idx = np.argmax(region)
                        local_y, local_x = np.unravel_index(flat_idx, region.shape)
                        
                        # Translate back to global tensor coordinates
                        global_y = start_y + local_y
                        global_x = start_x + local_x
                        
                        self.max_indices.append((b_idx, c_idx, global_y, global_x, i, j))
                        
        return output_tensor

    def backward(self, error_tensor):
        # We start with a zero gradient for the whole previous layer block
        error_prev = np.zeros(self.input_shape)
        
        # Distribute the error exclusively to the tracked maximal locations.
        # += accumulates gradients correctly when stride regions overlap.
        for b_idx, c_idx, global_y, global_x, i, j in self.max_indices:
            error_prev[b_idx, c_idx, global_y, global_x] += error_tensor[b_idx, c_idx, i, j]
            
        return error_prev