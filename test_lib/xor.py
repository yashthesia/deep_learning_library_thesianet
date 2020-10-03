from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y  = np.array([[0],[1],[1],[0]])

net = NeuralNet([
    Linear(2,10),
    Activation("relu"),
    Linear(10, 1),
    Activation("sigmoid")

])
train(net, X, y, epochs=50000, batch_size=32, optimizer=Adagrad(lr=0.001))
print(net.forward(X))
loss = BCE()
print(loss.loss(net.forward(X), y))


