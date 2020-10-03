"""
loss function is used to adjust the weights of the model
"""
from thesiaNet.tensor import tensor
import numpy as np

class MSE:
    def loss(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: MSE error
        """
        return np.sum((predicted - actual)**2)


    def grad(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: partial diff of MSE
        """
        return 2*(predicted-actual)


class BCE:
    def loss(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: binary cross entropy error
        """
        # print(np.mean((predicted - actual)**2)*0.5)
        return  np.sum(actual*np.log(predicted) + (1-actual)*np.log(1-predicted))*-1


    def grad(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: partial diff of BCE
        """
        return ((actual/predicted) - (1-actual)/(1-predicted))*-1



class CCE:
    def loss(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: categorical cross entropy
        """
        return sum(-1*actual*np.log(predicted))


    def grad(self, predicted, actual):
        """
        :param predicted: output of the forward pass of the NN
        :param actual: actual dataset's output
        :return: partial diff of MCE
        """
        return  -1*actual/predicted


