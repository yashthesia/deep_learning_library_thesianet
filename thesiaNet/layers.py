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
        self.params["w"] = np.random.rand(input_size, output_size)
        self.params["w"] = self.params["w"]/self.params["w"].sum()
        self.params["b"] = np.random.rand(output_size)
        self.params["b"] = self.params["b"] / self.params["b"].sum()
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
        self.params["activation"] = self.activation_name

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
            out = out / (out.sum(axis=0) + 1e-10)
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
            out = 1 / (1 + np.exp(self.input*-1))
            out = out * (1 - out)

        if self.activation_name == "linear":
            out = 1

        self.grads["activation"] = out * grads
        return out * grads
