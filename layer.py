import numpy as np
from paramopetations import ParamOperations
from functions import *

class Layer:

    def __init__(self, neurons):
        self.neurons = neurons
        self.first = True
        self.params = []
        self.param_grads = []
        self.operations = []
    
    def _setup_layer(self, num_in):
        raise NotImplementedError()
    
    def forward(self, input_):

        if self.first:
            self._setup_layer(input_)
            self.first = False
        
        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output
    
    def backward(self, output_grad):
        assert output_grad.shape == self.output.shape

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grad()

        return input_grad
    
    def _param_grad(self):
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperations):
                self.param_grads.append(operation.param_grad)
    
    def _params(self):
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperations):
                self.params.append(operation.param)


class Dense(Layer):

    def __init__(self, neurons, activation = Sigmoid):
        super().__init__(neurons)
        self.activation = activation
    
    def _setup_layer(self, input_):
        if self.seed:
            np.random.seed(self.seed)

        self.params= []

        # Weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # Bias
        self.params.append(np.random.randn(1, self.neurons))

        # Functions / Operations on Dense layyer
        self.operations = [
            WeightedMultiply(self.params[0]),
            BiasAddition(self.params[1]),
            self.activation()
        ]

        return None
    