from operations import Operations

class ParamOperations(Operations):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def backward(self, output_grad):
        
        assert output_grad.shape == self.output.shape

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_grad.shape == self.input.shape
        assert self.param_grad.shape ==self.param.shape

        return self.input_grad
    
    def _param_grad(self, output_grad):

        raise NotImplementedError()