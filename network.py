class NeuralNetwork:
    def __init__(self, layers, rate) -> None:
        self.alpha_ = rate
        self.layers_ = layers

    def forward(self, x):
        out = [x]
        for layer in self.layers_:
            out.append(layer.forward(out[-1]))
        return out
    
    def backward(self, outputs, y_real):
        y = outputs[-1]
        n = y_real.shape[0]
        mse = y_real - y
        mse = (mse * mse).sum() / n

        de = 2 / n * (y - y_real)

        for i in reversed(range(len(self.layers_))):
            layer = self.layers_[i]
            x = outputs[i]
            de = layer.backward(de, x, self.alpha_)
        return mse
    
    def train(self, x_set, y_set):
        assert len(x_set) == len(y_set)

        for i in range(len(x_set)):
            y_real = y_set[i]
            x = x_set[i]
            
            outputs = self.forward(x)
            print(outputs)
            print(f"MSE: {self.backward(outputs, y_real)};{' ' * 20}Learning rate: {self.alpha_}")

        # for layer in self.layers_:
        #     print(layer.w_)
        return True
    
    def predict(self, x):
        current = x
        for layer in self.layers_:
            current = layer.forward(current)
        return current
    