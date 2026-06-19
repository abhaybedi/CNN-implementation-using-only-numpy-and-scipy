import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._current_label_tensor = None

    def forward(self):
        # Fetch data
        input_tensor, label_tensor = self.data_layer.next()
        self._current_label_tensor = label_tensor
        
        # Forward pass through the network layers
        activation = input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)
            
        # Get loss
        return self.loss_layer.forward(activation, self._current_label_tensor)

    def backward(self):
        # Initiate backward pass from the loss layer
        error = self.loss_layer.backward(self._current_label_tensor)
        
        # Propagate error backward through the network layers
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        # Assign a deep copy of the optimizer to trainable layers
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        # Run training loop
        for _ in range(iterations):
            current_loss = self.forward()
            self.loss.append(current_loss)
            self.backward()

    def test(self, input_tensor):
        # Forward pass for testing/inference without invoking loss layer
        activation = input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation