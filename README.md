# CNN-implementation-using-only-numpy-and-scipy
from-scratch implementation of key CNN building blocks as part of a deep learning framework: Xavier/He initialization, optimizers: SGD with Momentum, Adam and essential layers Convolution, Max-Pooling, Flatten, fully compatible with a custom neural network pipeline.   No Tensorflow/Pytorch used


# CNN From Scratch

This project is part of a deep learning exercise where the goal was to implement the core building blocks of Convolutional Neural Networks (CNNs) completely from scratch — no TensorFlow, no PyTorch, just NumPy and logic.

The focus here is on understanding what’s really going on under the hood: forward passes, backpropagation, weight initialization, and optimization.

## What’s implemented

### Initializers
I implemented a few common initialization strategies:
- Constant
- Uniform random
- Xavier (Glorot)
- He (Kaiming)

These are used to properly initialize weights depending on the network setup.

### Optimizers
Basic SGD was already there, and I added:
- SGD with Momentum
- Adam

Both update parameters per layer and keep internal state (like velocity or moments).

### Layers
The following layers are implemented:

- **Fully Connected (extended)**  
  Now supports custom weight and bias initialization.

- **Convolution Layer**
  - Works for both 1D and 2D inputs  
  - Uses "same" padding  
  - Supports different stride configurations  
  - Includes full backward pass (gradients for weights, bias, and input)

- **Pooling Layer**
  - Max pooling (2D only)
  - Uses "valid" padding (no zero padding)
  - Keeps track of max locations for backprop

- **Flatten Layer**
  - Converts multi-dimensional input into a vector
  - Restores original shape during backward pass
