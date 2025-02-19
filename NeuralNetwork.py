import numpy as np
from layer import Dense
from loss import Loss

class NeuralNetwork:

    def __init__(self, layers, loss, seed = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)
    
    def forward(self, x_batch):
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        
        return x_out

    def backward(self, loss_grad):

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return None

    def train_batch(self, x_batch, y_batch):
        prediction = self.forward(x_batch)
        loss = self.loss.forward(prediction, y_batch)
        self.backward(self.loss.backward())
        return loss
    
    def params(self):
        for layer in self.layers:
            yield from layer.params
    
    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads