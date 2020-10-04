from sklearn.datasets import load_digits
from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np
import matplotlib.pylab as plt

# load the dataset with 10 classes
np.random.seed(2401998)

digits = load_digits(n_class=2)
# print(digits.data.shape)
# print(digits.target.shape)

#data Normalization
X = (digits.data/16).reshape(-1,8,8,1)
y = digits.target.reshape(-1,1)

X_train, X_test, y_train, y_test = X[:350], X[350:], y[:350], y[350:]
print(y_test)
# print(X.shape)
# print(y.shape)

plt.imshow(X[0,:,:,0])
plt.show()


# define NN
net = NeuralNet([
    Convo(
       input_size=(8,8,1),
       filter_size=3,
       number_of_filters= 16,
       padding= 0,
       stride=1
    ),
    Activation('relu'),
    Convo(
       input_size=(6,6,16),
       filter_size=3,
       number_of_filters= 32,
       padding= 0,
       stride=1
    ),
    Activation('relu'),
    Flatten((4,4,32)),
    Linear(input_size=512, output_size= 32),
    Activation('relu'),
    Linear(input_size=32, output_size=1),
    Activation('sigmoid')

])

for i in range(20):


    y_pred = net.forward(X_test)
    acc = 0
    for i in range(y_test.shape[0]):
        pred = 0
        if y_pred[i,0]>0.5:
            pred = 1
        if pred == y_test[i,0]:
            acc+=1

    print(f'accuracy : {acc/y_test.shape[0]}')

    train(net, X_train, y_train, epochs=1, batch_size=1000, optimizer=SGD(0.0005), loss=BCE()

"""
accuracy : 0.6
epoch 1 | Loss : 279.57923890986206
accuracy : 0.4
epoch 1 | Loss : 194.79309884669067
accuracy : 1.0
epoch 1 | Loss : 122.88819386215138
accuracy : 1.0
epoch 1 | Loss : 73.851960306665
accuracy : 1.0
epoch 1 | Loss : 52.23603323968721
accuracy : 1.0
epoch 1 | Loss : 38.92734886316832
accuracy : 1.0
epoch 1 | Loss : 30.49420362755029
accuracy : 1.0
epoch 1 | Loss : 24.808540078694833
accuracy : 1.0
epoch 1 | Loss : 20.77008432828483
accuracy : 1.0
"""