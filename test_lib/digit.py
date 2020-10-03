from sklearn.datasets import load_digits
from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np


# load the dataset with 10 classes
np.random.seed(2401998)

digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)

#data Normalization
X = digits.data/16
y = digits.target


#one hot encoding
yy = []
for i in range(len(y)):
    classes = np.zeros(10)
    classes[y[i]] = 1
    yy.append(classes)

yy = np.array(yy)
print(yy.shape)


# define NN
net = NeuralNet([
    Linear(64,128),
    Activation("tanh"),
    Linear(128,256),
    Activation("tanh"),
    Linear(256,10),
    Activation("softmax")
])

train(net, X, yy, epochs=3000, batch_size=1000, optimizer= SGD(), loss = CCE())

loss = CCE()
print(loss.loss(yy,net.forward(X)))
