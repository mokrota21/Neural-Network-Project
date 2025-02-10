class Layer:
    """
    Base class for any layer in neural network. Expects init, forward and backward to be defined in descendant. Represents next layer,
    for instance if we have layer 1 and 2 (input 1, output 2), instance of Layer that stores weights between 1 and 2 would represent layer 2.
    """
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass