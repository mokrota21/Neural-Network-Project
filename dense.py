import numpy as np

class DenseLayer():
    def __init__(self, input_size, output_size) -> None:
        self.w_ = np.random.random((output_size, input_size))
        self.b_ = np.random.random((output_size, 1))
    
    def forward(self, input):
        out = np.matmul(self.w_, input) + self.b_
        return out

    def backward(self, de, x, rate):
        n, m = self.w_.shape
        
        # gradient descent for weight
        for i in range(n):
            for j in range(m):
                w_prime = de[i, 0] * x[j, 0]
                # print(w_prime)
                self.w_[i, j] -= w_prime * rate

        # gradient descent for bias
        for i in range(n):
            self.b_ -= de[i, 0] * rate
        
        dx = np.zeros((m, 1), dtype=float)

        for i in range(m):
            for j in range(n):
                dx[i, 0] += de[j, 0] * self.w_[j, i]
        
        return dx
