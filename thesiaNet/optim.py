from thesiaNet.tensor import tensor
import thesiaNet.layers
import numpy as np



class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def step(self, net):
        for param, grad in net.params_and_grads():
            param -= self.lr*grad


class Momentum:
    def __init__(self, lr = 0.01, decay = 0.99):
        self.lr = lr
        self.decay = decay
        self.v = []
        self.cur_step = 0

    def step(self, net):
        i = 0
        for param, grad in net.params_and_grads():
            if self.cur_step == 0:
                self.v.append(self.lr*grad)
            else:
                self.v[i] = self.decay*self.v[i] + self.lr*grad

            param -= self.v[i]
            i+=1

        self.cur_step +=1


class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.catche = []
        self.cur_step = 0

    def step(self, net):
        i = 0
        for param, grad in net.params_and_grads():
            if self.cur_step == 0:
                self.catche.append(grad**2)
            else:
                self.catche[i] =  grad**2

            param -= self.lr * grad /np.sqrt(self.catche[i] + 1e-7)
            i+=1

        self.cur_step +=1