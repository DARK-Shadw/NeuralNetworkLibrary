class Optimizer:
    def __init__(self, lr):
        self.lr = lr
    
    def step(self):
        pass

class SGD(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)
    
    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad