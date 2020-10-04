import numpy as np


class Flatten:
    def __init__(self, input_shape = None):
        self.params = {}
        self.grads= {}
        self.params['not_trainable'] = 'Fatten'
        self.input_shape = input_shape
        self.out_shape = 1
        for f in self.input_shape:
            self.out_shape *= f

    def forward(self, input):
        return input.reshape(-1,self.out_shape)

    def backward(self, grad):
        self.grads['not_trainable'] =  np.reshape(grad,tuple([-1]+list(self.input_shape)))
        return self.grads['not_trainable']






x = np.random.randn(1000,50,50,1)
layer = Flatten((50,50,1))
print(layer.forward(x).shape)
print(layer.backward(layer.forward(x)).shape)