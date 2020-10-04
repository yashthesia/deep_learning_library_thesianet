"""
List of Deep learning layers
Example :
input -> linear -> activation -> output
"""

import numpy as np
from thesiaNet.tensor import tensor


class Linear:
    def __init__(self, input_size, output_size):
        """
        :param input_size: input size
        :param output_size: output size
        """
        self.input_size = input_size
        self.output_size = output_size

        # initialization of the layer parameter
        self.params = {}
        self.grads = {}
        self.params["w"] = np.random.randn(input_size, output_size)/np.sqrt(input_size/2) # here we are using xavier initialization
        self.params["b"] = np.random.rand(output_size)/np.sqrt(input_size/2)
        return

    def forward(self, input):
        """
        :param input: data from the direct input or previous layer
        :return:  forward pass of the Liner layer
        """
        self.input = input

        return  self.input @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        """
        if y = f(x) and x = input@w+b
        dy/dinput = f'(x)*w.T
        dy/dw = input.T*f'(x)
        dy/db = f'(x)

        :param grad: g'(z) which we need to backward pass through layer
        :return:
        """


        self.grads["input"] = grad @ self.params["w"].T
        self.grads["w"] = self.input.T @ grad
        self.grads["b"] = grad.sum(axis = 0)

        return self.grads["input"]





class Activation:
    def __init__(self, activation_name):
        self.activation_name = activation_name
        self.params = {}
        self.grads = {}
        self.params["not_trainable"] = self.activation_name

    def forward(self, input):
        """
        :param input: data from the direct input or previous layer
        :return:  forward pass of the Liner layer
        """
        self.input = input
        if self.activation_name == "tanh":
            out = np.tanh(self.input)
            return out

        if self.activation_name == "relu":
            out = np.maximum(self.input,0)
            return out

        if self.activation_name == "sigmoid":
            out = 1 / (1 + np.exp(self.input*-1))
            return out

        if self.activation_name == "linear":
            out = self.input
            return out


        if self.activation_name == "softmax":
            out = np.exp( self.input)
            out = out / (out.sum(axis=-1).reshape(-1,1) + 1e-10)
            return out



    def backward(self, grads):
        """
        :param grads: partial derivative from previous layer
        :return: output after passing gradient from  the given layer
        """

        if self.activation_name == "tanh":
            out = 1 - np.tanh(self.input)**2


        if self.activation_name == "relu":
            out = np.sign(np.maximum(self.input,0))

        if self.activation_name == "sigmoid":
            out = 1 / (1 + np.exp(self.input*-1))
            out = out * (1 - out)

        if self.activation_name == "softmax":
            grads = grads - ((grads*self.forward(self.input)).sum(axis = -1)).reshape(-1,1)
            out = self.forward(self.input)

        if self.activation_name == "linear":
            out = 1


        self.grads["not_trainable"] = out * grads
        return out * grads





class Convo:
    def __init__(self,
                 input_size,
                 filter_size=3,
                 number_of_filters=64,
                 padding=0,
                 stride=1):
        """
        This class apply convolution 2d on inout data
        here I assume that the input dimension would be : (Batch size, Height, Widhth, Features)
        :param input_size: input dimension
        :param filter_size: filter size (assuming that the shape is square)
        :param number_of_filters: output features or number of filters
        :param padding: padding applied in Height and width
        :param stride: stride
        """
        self.input_size = input_size  # (H, W, D)
        # print(input_size)
        self.H_in, self.W_in, self.D_in = self.input_size
        self.D_out = number_of_filters

        self.filer_size = filter_size
        self.padding = padding
        self.stride = stride

        self.params = {}
        self.grads = {}
        self.params['w'] = np.random.randn(self.filer_size, self.filer_size, self.D_in, self.D_out)/np.sqrt(self.H_in*self.W_in*0.5)
        self.params['b'] = np.random.randn(self.D_out)/np.sqrt(self.H_in*self.W_in*0.5)


    def forward(self, input):
        # save the input for gradient calculation and add padding on it
        self.input = np.pad(input, [(0, 0), (self.padding,self.padding),(self.padding,self.padding),(0,0)], mode='constant', constant_values=0)

        #calculating output dimensions
        self.batch, _, _, _ = input.shape
        self.H_out = ((self.H_in + 2*self.padding -self.filer_size)//(self.stride)) +1
        self.W_out = ((self.W_in + 2 * self.padding - self.filer_size) // (self.stride)) + 1

        out = np.random.randn(self.batch, self.H_out, self.W_out, self.D_out)*0

        for m in range(self.batch):
           for i in range(0,self.input.shape[1]-self.filer_size+1,self.stride):
               for j in range(0,self.input.shape[2]-self.filer_size+1,self.stride):
                   for f in range(self.D_out):
                       out[m,i,j,f] = np.sum(np.multiply(self.input[m,i:i+self.filer_size,j:j+self.filer_size,:],self.params['w'][:,:,:,f])) + self.params['b'][f]

        return out


    def backward(self, grad):
        self.grads['w'] = np.random.randn(self.filer_size, self.filer_size, self.D_in, self.D_out)*0
        self.grads['b'] = np.random.randn(self.D_out)*0
        self.grads['input'] = np.random.randn(self.input.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3]) * 0


        for m in range(self.batch):
            for i in range(0, self.input.shape[1] - self.filer_size + 1, self.stride):
                for j in range(0, self.input.shape[2] - self.filer_size + 1, self.stride):
                    for f in range(self.D_out):
                        self.grads['input'][m,i:i+self.filer_size,j:j+self.filer_size,:] += self.params['w'][:,:,:,f]*grad[m,i,j,f]
                        self.grads['w'][:,:,:,f] += self.input[m,i:i+self.filer_size,j:j+self.filer_size,:]*grad[m,i,j,f]
                        self.grads['b'][f] = grad[m,i,j,f]


        if self.padding > 0:
            self.grads['input'] = self.grads['input'][:,self.padding:-self.padding,self.padding:-self.padding]


        return self.grads['input']


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
        self.grads['not_trainable'] =  grad.reshape(tuple([-1]+list(self.input_shape)))

        return self.grads['not_trainable']

