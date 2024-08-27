import numpy as np

class DenseLayer():
    def __init__(self, input_size, output_size) -> None:
        self.x_ = None
        self.w_ = np.zeros((output_size, input_size), dtype=float)
        self.b_ = np.zeros((output_size, 1), dtype=float)
    
    def forward(self, input):
        out = np.matmul(self.w_, input) + self.b_
        self.x_ = input
        return out

    def backward(self, de, rate):
        n, m = self.w_.shape
        
        # gradient descent for weight
        for i in range(n):
            for j in range(m):
                w_prime = de[i, 0] * self.x_[j, 0]
                self.w_[i, j] -= w_prime * rate
        
        dx = np.zeros((m, 1), dtype=float)

        for i in range(m):
            for j in range(n):
                dx[i, 0] += de[j, 0] * self.w_[j, i]
        
        return dx

                
