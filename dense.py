import numpy as np
from random import uniform
from layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size, output_size) -> None:
        self.w_ = np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size + output_size))
        self.b_ = np.zeros(output_size)
    
    def forward(self, input):
        out = np.dot(self.w_, input) + self.b_
        return out
    
    def derrivative(self, de):
        return np.matmul(self.w_.T, de)

    def backward(self, de, x, rate, gamma):
        dx = self.derrivative(de) 
        
        # gradient descent for weight
        self.w_ -= np.matmul(de, x.T) * rate

        # gradient descent for bias
        self.b_ -= de * rate
        
        return dx
