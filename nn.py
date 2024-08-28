class NeuralNetwork:
    def __init__(self, layers, rate) -> None:
        self.alpha_ = rate
        self.layers_ = layers
    
    def forward (self, x):
        outputs = [x]
        for layer in self.layers_:
            outputs.append(layer.forward(outputs[-1]))
        return outputs

    def backward(self, outputs, y, y_real):
        mse = 0
        de = 0
        n = y.shape[0]

        mse = ((y_real - y) * (y_real - y)).sum() / n
        de = 2 * (y - y_real) / n

        for i in reversed(range(len(self.layers_))):
            layer = self.layers_[i]
            x = outputs[i]
            print(f"Error derrivative: {de}")
            de = layer.backward(de, x, self.alpha_)

        # for layer in self.layers_[::-1]:
        #     de = layer.backward(de, x, self.alpha_)
        
        return mse

    def train(self, x_set, y_set):
        # every input has output
        assert len(x_set) == len(y_set)

        for i in range(len(x_set)):
            y_real = y_set[i]
            x = x_set[i]
            
            outputs = self.forward(x)
            y = outputs[-1]
            print(f"MSE: {self.backward(outputs, y, y_real)};{' ' * 20}Learning rate: {self.alpha_}")

        return True

    def predict(self, x):
        current = x
        for layer in self.layers_:
            current = layer.forward(current)
        return current
        