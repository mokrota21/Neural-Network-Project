from math import exp
import numpy as np

class ActivationLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, de, x, rate):
        return de * (1 - np.tanh(x) ** 2)