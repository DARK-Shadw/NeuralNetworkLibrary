import numpy as np
from paramopetations import ParamOperations
from operations import Operations

class WeightedMultiply(ParamOperations):

    def __init__(self, W):
        super().__init__(W)

    def _output(self):
        return np.dot(self.input, self.param)
    
    def _input_grad(self, output_grad):
        return np.dot(output_grad, np.transpose(self.param, (1,0)))
    
    def _param_grad(self, output_grad):
        return np.dot(np.transpose(self.input, (1, 0)), output_grad)

class BiasAddition(ParamOperations):

    def __init__(self, B):
        super().__init__(B)

    def _output(self):
        return self.input + self.param
    
    def _input_grad(self, output_grad):
        return np.ones_like(self.input) * output_grad
    
    def _param_grad(self, output_grad):
        param_grad = np.ones_like(self.param) * output_grad

        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
    
class Sigmoid(Operations):

    def __init__(self):
        super().__init__()

    def _output(self):
        return 1.0/(1.0 + np.exp(-1.0 * self.input))
    
    def _input_grad(self, output_grad):
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward *output_grad
        return input_grad
    
class Softmax:
    def __init__(self):
        pass
    
    def forward(self, input_):
        # To avoid overflow, subtract the max value from the input.
        # This doesn't affect the output as it's a constant shift.
        exps = np.exp(input_ - np.max(input_, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, output_grad):
        # The backward pass requires the Jacobian of the Softmax function
        # For simplicity, we will compute the gradient using the following formula
        # for softmax:
        #   dL/dz = softmax_output - y_true (one-hot encoded)
        
        return output_grad
