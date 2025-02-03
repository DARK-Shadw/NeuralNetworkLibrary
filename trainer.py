import numpy as np
from copy import deepcopy
from NeuralNetwork import NeuralNetwork
from optimizer import Optimizer
from utils import *

class Trainer:
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        setattr(self.optim, "net", self.net)
    
    def fit(self, X_train, y_train, X_test, Y_test, epochs = 100, eval_every = 10, batch_size = 32, seed = 1, restart = True):
        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        for e in range(epochs):
            if (e+1) % eval_every == 0:
                last_model = deepcopy(self.net)
            
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = generate_batch(X_train, y_train, batch_size)

            for ii, (X_batch, Y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, Y_batch)
                self.optim.step()
            
            if (e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, Y_test)

                if loss < self.best_loss:
                    print(f"Validation Loss after {e+1} epochs is {loss:.3f}")
                else:
                    print(f"Loss increased after epochs {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}")
                    self.net = last_model
                    setattr(self.optim, "net", self.net)