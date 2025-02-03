import numpy as np

class Loss:

    def __init__(self):
        pass

    def forward(self, prediction, target):
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value
    
    def backward(self):
        self.input_grad = self._input_grad()

        assert self.input_grad.shape == self.prediction.shape

        return self.input_grad
    
    def _output(self):
        raise NotImplementedError()
    
    def _input_grad(self):
        raise NotImplementedError()
    

class MeanSquaredError(Loss):

    def __init__(self):
        super().__init__()

    def _output(self):
        loss = np.sum(np.power(self.prediction - self.target, 2))/ self.prediction.shape[0]

        return loss
    
    def _input_grad(self):
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
    
class CrossEntropyLoss(Loss):

    def __init__(self):
        super().__init__()

    def _output(self):
        m = self.prediction.shape[0]
        log_probs = -np.log(self.prediction[range(m), self.target])
        loss = np.sum(log_probs) / m
        return loss
    
    def _input_grad(self):
        m = self.prediction.shape[0]
        grad = self.prediction.copy()
        grad[range(m), self.target] -= 1
        return grad / m
