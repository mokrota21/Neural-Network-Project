import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size) -> None:
        self.w_ = np.random.random((output_size, input_size))
        self.b_ = np.random.random((output_size, 1))
    
    def forward(self, input):
        out = np.matmul(self.w_, input) + self.b_
        return out

    def backward(self, de, x, rate):
        # gradient descent for weight
        self.w_ -= np.matmul(de, x.T) * rate

        # gradient descent for bias
        self.b_ -= de * rate
        
        dx = np.matmul(self.w_.T, de)
        
        return dx
