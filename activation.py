from math import tanh, log, exp
import numpy as np

def tanh_prime(x):
    return 1 - tanh(x) ** 2

def softPlus(x):
    return log(1 + exp(x))

def softPlus_prime(x):
    return 1 / (1 + exp(-1 * x))

def softPlus_np(x):
    return np.log1p(x)

def softPlus_np_prime(x):
    return 1.0 / (1 + np.exp(-1 * x))

class ActivationLayer:
    def __init__(self, activation, activation_prime, vectorized=False) -> None:
        if not vectorized:
            activation = np.vectorize(activation)
            activation_prime_ = np.vectorize(activation_prime)

        self.activation_ = activation
        self.activation_prime_ = activation_prime

    def forward(self, x):
        return self.activation_(x)
    
    def backward(self, de, x, rate):
        return de * self.activation_prime_(x)