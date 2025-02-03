import numpy as np
class Operations:
    # A base class for operations will define functions required for basic operations
    # A basic operation is stuff like dot product, addition, sigmoid etc..
    # So the class can include stuff like output, grad, forward and backward fucntions
    # The forward function will be the public function which will do stuff like saving input value
    # The forward function will then call the output function
    # The backward function will be also the public fucntion for finding the gradient
    # The ouput function can do the actuial operation and return the result
    # The grad fulction can find the gradient based on the recieved output gradient
    
    def __init__(self):
        pass

    def forward(self, input_):
        
        self.input = input_

        self.output = self._output()
        
        return self.output
    
    def backward(self, output_grad):

        assert output_grad.shape == self.output.shape

        self.input_grad = self._input_grad(output_grad)

        assert self.input_grad.shape == self.input.shape

        return self.input_grad
    
    def _output(self):

        raise NotImplementedError()
    
    def _input_grad(self, output_grad):

        raise NotImplementedError()