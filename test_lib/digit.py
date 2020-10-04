from sklearn.datasets import load_digits
from thesiaNet.train import *
from thesiaNet.nn import *
from thesiaNet.layers import *
import numpy as np
import matplotlib.pylab as plt

# load the dataset with 10 classes
np.random.seed(2401998)

digits = load_digits(n_class=10)
# print(digits.data.shape)
# print(digits.target.shape)

#data Normalization
X = (digits.data/16)
y = digits.target


yy = []
for i in range(len(y)):
    classes = np.zeros(10)
    classes[y[i]] = 1
    yy.append(classes)

yy = np.array(yy)
print(yy.shape)
y = yy

X_train, X_test, y_train, y_test = X[:1700], X[1700:], y[:1700], y[1700:]


# define NN
net = NeuralNet([
    Linear(64,128),
    Activation("relu"),
    Linear(128,256),
    Activation("relu"),
    Linear(256,10),
    Activation("softmax")
])



def acc(pred, actual):
    pred = np.argmax(pred, axis=-1)
    actual = np.argmax(actual, axis=-1)
    return np.where(pred == actual)[0].shape[0] / (len(pred))

loss = MSE()

test_acc  = []
train_acc  = []
# print(acc(net.forward(X_train),y_train))
for i in range(2000):
    train(net, X_train, y_train, epochs=1, batch_size=1000, optimizer=Adagrad(0.001), loss=MSE(), epoch_skip=100)
    train_acc.append([acc(net.forward(X_train),y_train)])
    test_acc.append([acc(net.forward(X_test), y_test)])

plt.plot(train_acc)
plt.plot(test_acc)
plt.show()




"""
epoch 2995 | Loss : 0.00023471123743766173
epoch 2996 | Loss : 0.0002344296019998131
epoch 2997 | Loss : 0.00023414436918530482
epoch 2998 | Loss : 0.0002338611273334355
epoch 2999 | Loss : 0.00023357278455604698
epoch 3000 | Loss : 0.00023328288289475947


[[1.00000000e+00 6.67276292e-28 2.04865147e-19 ... 1.01235855e-14
  3.36541312e-17 2.33887712e-14]
 [1.23326892e-23 1.00000000e+00 9.99220741e-23 ... 3.94272081e-20
  1.78714775e-15 4.20293790e-24]
 [8.10928916e-15 2.16020659e-08 9.99999827e-01 ... 3.79800409e-17
  1.51417152e-07 1.85103336e-22]
 ...
 [6.09963706e-19 1.46411112e-16 8.68424389e-22 ... 2.66249255e-25
  1.00000000e+00 4.49531464e-23]
 [1.65932872e-13 7.56026192e-22 2.88710343e-26 ... 5.94523215e-20
  3.89597508e-12 1.00000000e+00]
 [3.09831856e-17 1.34573571e-18 2.84425295e-21 ... 5.63191305e-25
  1.00000000e+00 2.97828672e-14]]

"""