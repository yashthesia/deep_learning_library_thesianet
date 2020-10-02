from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y  = np.array([[0],[1],[1],[0]])

net = NeuralNet([
    Linear(2,100),
    Activation("relu"),
    Linear(100, 100),
    Activation("relu"),
    Linear(100, 1),
    Activation("sigmoid")

])
train(net, X, y, epochs=500000, batch_size=32)
print(net.forward(X))
loss = BCE()
print(loss.loss(net.forward(X), y))


