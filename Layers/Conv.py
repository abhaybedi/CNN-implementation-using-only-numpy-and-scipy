import numpy as np
import scipy.signal
import copy
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        
        # Standardize stride_shape to a tuple (handles 1D and 2D)
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = tuple(stride_shape)
            
        self.convolution_shape = tuple(convolution_shape)
        self.num_kernels = num_kernels
        self.is_1d = len(self.stride_shape) == 1

        # Initialize parameters uniformly in [0, 1)
        self.weights = np.random.uniform(0, 1, (num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None

        self._optimizer = None
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        # We need two distinct copies of the optimizer to handle weights and bias separately
        self._optimizer = copy.deepcopy(opt)
        self._bias_optimizer = copy.deepcopy(opt)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # Calculate expected output spatial dimensions
        output_shape = []
        for i in range(len(self.stride_shape)):
            output_shape.append(int(np.ceil(input_tensor.shape[2 + i] / self.stride_shape[i])))
        
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_shape))

        # Perform cross-correlation
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(self.convolution_shape[0]):
                    # Mode 'same' ensures spatial dimensions remain intact prior to striding
                    corr = scipy.signal.correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                    
                    # Subsample based on stride shape
                    if self.is_1d:
                        output_tensor[b, k] += corr[::self.stride_shape[0]]
                    else:
                        output_tensor[b, k] += corr[::self.stride_shape[0], ::self.stride_shape[1]]
                        
                # Add the specific kernel's bias
                output_tensor[b, k] += self.bias[k]

        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        
        # 1. Upsample the error tensor (injecting zeros based on stride)
        upsampled_shape = list(error_tensor.shape)
        for i in range(len(self.stride_shape)):
            upsampled_shape[2 + i] = self.input_tensor.shape[2 + i]
        
        upsampled_error = np.zeros(upsampled_shape)
        
        if self.is_1d:
            upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor
        else:
            upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        # 2. Gradient w.r.t Bias (Summing over batches and spatial dims)
        axes_to_sum = tuple([0] + list(range(2, len(error_tensor.shape))))
        self.gradient_bias = np.sum(error_tensor, axis=axes_to_sum)

        # 3. Prepare input padding to compute valid weight gradients
        pad_widths = []
        for i in range(len(self.convolution_shape) - 1):
            kernel_dim = self.convolution_shape[i + 1]
            pad_front = kernel_dim // 2
            pad_back = (kernel_dim - 1) // 2
            pad_widths.append((pad_front, pad_back))
            
        # Pad batch and channels with 0, pad spatial axes dynamically
        spatial_pad = [(0, 0), (0, 0)] + pad_widths
        padded_input = np.pad(self.input_tensor, spatial_pad, mode='constant')

        self.gradient_weights = np.zeros_like(self.weights)
        error_tensor_prev = np.zeros_like(self.input_tensor)

        # 4. Compute Weight Gradients and Propagate Error downwards
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(self.convolution_shape[0]):
                    # Valid correlation over padded input yields correct gradient size
                    grad = scipy.signal.correlate(padded_input[b, c], upsampled_error[b, k], mode='valid')
                    self.gradient_weights[k, c] += grad
                    
                    # Convolving the upsampled error with weights sends the error back
                    err_prev = scipy.signal.convolve(upsampled_error[b, k], self.weights[k, c], mode='same')
                    error_tensor_prev[b, c] += err_prev

        # 5. Parameter updates
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return error_tensor_prev

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)