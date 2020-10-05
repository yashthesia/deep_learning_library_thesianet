from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np

# this file is to test the convolutional layer
# here the input image is simple image with random noise in it
# output: sum of left upper corner pixels value

X = np.random.randn(1000, 10, 10, 3)
y = X[:, :, :, 0:1] + X[:, :, :, 1:2] + X[:, :, :, 2:3]
y = y[:, :-2, :-2]

net = NeuralNet([
    Convo((10, 10, 3), number_of_filters=1)
])
train(net, X, y, epochs=200, batch_size=1000, optimizer=Adagrad(lr=0.001))

# testing data
X = np.random.randn(1, 10, 10, 3)
y = X[:, :, :, 0:1] + X[:, :, :, 1:2] + X[:, :, :, 2:3]
y = y[:, :-2, :-2]
print(np.mean(net.forward(X) - y))
