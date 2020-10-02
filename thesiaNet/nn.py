from thesiaNet.tensor import tensor
import numpy as np
import thesiaNet.layers

class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        """
        this function forward pass the entire data batch and gives the prediction
        :param input: original input
        :return: predicted value
        """
        self.input = input

        for layer in self.layers:
            input = layer.forward(input)
            # print(input.shape)

        return input

    def backward(self, grad):
        """
        this function calculates and saves the gradient values of each layer
        :param grad: partial derivative of the loss function
        :return: return the gradiennt values after passing the entire network
        """

        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
            # print("pass" , grad.shape)
        return grad

    def params_and_grads(self):
        pass_data = []
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                if name != "activation":
                    pass_data.append((param, grad))

        return pass_data



