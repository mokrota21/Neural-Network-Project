from random import randint
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, rate, threshold=0.0001) -> None:
        self.alpha_ = rate
        self.layers_ = layers
        self.best_ = None
        self.last_mse_ = -1
        self.count = 0
        self.threshold_ = threshold

    def forward(self, x):
        out = [x]
        for layer in self.layers_:
            out.append(layer.forward(out[-1]))
        return out
    
    def backward(self, outputs, y_real):
        # print(outputs)
        y = outputs[-1]
        n = y_real.shape[0]
        mse = y_real - y
        # print(y_real, y, n)
        mse = (mse * mse).sum() / n

        de = 2 / n * (y - y_real)

        for i in reversed(range(len(self.layers_))):
            layer = self.layers_[i]
            x = outputs[i]
            de = layer.backward(de, x, self.alpha_)
        return mse
    
    def randomize_dense(self):
        for layer in self.layers_:
            try:
                layer.w_ = layer.w_ * (np.random.random(layer.w_.shape) * 2 - 1)
                layer.b_ = layer.b_ * (np.random.random(layer.b_.shape) * 2 - 1)
            except Exception:
                pass
        return True
    
    def train_util(self, x_set, y_set):
        avg_mse = 0

        for i in range(len(x_set)):
            y_real = y_set[i]
            x = x_set[i]
            
            outputs = self.forward(x)
            mse = self.backward(outputs, y_real)
            avg_mse += mse
        avg_mse = avg_mse / len(x_set)
        
        print_best = -1 if self.best_ is None else self.best_[0]
        print(f"New MSE: {avg_mse}, Best MSE: {print_best}")
        if (self.best_ is None or avg_mse < self.best_[0]):
            self.best_ = (avg_mse, self.layers_.copy())
    
    def too_big_alpha(self, mse):
        return self.last_mse_ - mse < 0
    
    def too_small_alpha(self, mse):
        return self.last_mse_ - mse < self.threshold_
    
    # When training we do random weights if local minimum achieved. 
    def train(self, x_set, y_set):
        assert len(x_set) == len(y_set)
        total_mse = 0
        count = 0

        for i in range(len(x_set)):
            y_real = y_set[i]
            x = x_set[i]
            
            outputs = self.forward(x)
            
            mse = self.backward(outputs, y_real)
            print(mse)
            total_mse += mse
            count += 1
            if self.too_big_alpha(mse):
                print("decrease 10 times")
                self.alpha_ /= 10
            elif self.too_small_alpha(mse):
                if self.count < 10:
                    print("increase 10 times")
                    self.alpha_ *= 10
                else:
                    if (self.best_ is None or total_mse / count < self.best_[0]):
                        self.best_ = (total_mse / count, self.layers_.copy())
                    self.randomize_dense()
                    self.count = 0
                    count = 0
                    total_mse = 0
            
            self.last_mse_ = mse

        total_mse = total_mse / len(x_set)

        self.layers_ = self.layers_ if self.best_ is None else self.best_[1]
        return total_mse if self.best_ is None else self.best_[0]
    
    def predict(self, x):
        current = x
        for layer in self.layers_:
            current = layer.forward(current)
        return current
        
    