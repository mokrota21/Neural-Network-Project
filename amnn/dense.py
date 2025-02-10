from layer import Layer
import numpy as np

class DenseLayer(Layer):
    """
    Dense Layer class. Represents fully connected layer.
    Args:
    n - number of neurons in input;
    m - number of neurons in output;
    """
    def __init__(self, n, m):
        self.weights = np.random.rand(n, m)
        self.output = np.empty(m)

    def forward(self, x: np.ndarray):
        assert x.shape == self.weights.shape[0]
        self.output = self.weights