class BaseLayer:
    def __init__(self):
        self.trainable = False #distinguish trainable from non-trainable layers
        self.weights = None
