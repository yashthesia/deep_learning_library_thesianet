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

# data Normalization
X = (digits.data / 16).reshape(-1, 8, 8, 1)
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

plt.imshow(X[0, :, :, 0])
plt.show()

# define NN
net = NeuralNet([
    Convo(
        input_size=(8, 8, 1),
        filter_size=3,
        number_of_filters=16,
        padding=0,
        stride=1
    ),
    Activation('relu'),
    Convo(
        input_size=(6, 6, 16),
        filter_size=3,
        number_of_filters=32,
        padding=0,
        stride=1
    ),
    Activation('relu'),
    Flatten((4, 4, 32)),
    Linear(input_size=512, output_size=32),
    Activation('relu'),
    Linear(input_size=32, output_size=10),
    Activation('softmax')

])

loss = CCE()

test_loss = []
for i in range(20):
    train(net, X_train, y_train, epochs=5, batch_size=1000, optimizer=Adagrad(0.001), loss=CCE(), epoch_skip=5)
    tl = loss.loss(net.forward(X_test), y_test)
    print(f"validatiion loss {tl}")
    test_loss.append([tl])

plt.plot(test_loss)
plt.show()

"""
epoch 1 | Loss : [0.10602534 0.03341162 0.02476159 0.04544231 0.00758518 0.06766345
 0.06801428 0.07378455 0.00068257 0.06808907]
epoch 2 | Loss : [0.09760016 0.0309499  0.02304907 0.04234874 0.0069901  0.062705
 0.06295571 0.06818762 0.00061421 0.06287963]
epoch 3 | Loss : [0.09035654 0.02894852 0.02152476 0.03961262 0.00649182 0.05832827
 0.05838295 0.06320258 0.00056141 0.05847883]
epoch 4 | Loss : [0.08403691 0.02717875 0.02018114 0.03719374 0.00605542 0.05447945
 0.05437966 0.05884028 0.0005159  0.05461085]
epoch 5 | Loss : [0.07847633 0.02560341 0.01898798 0.03504116 0.00567046 0.05107077
 0.05084859 0.05499422 0.00047635 0.05118797]
test error : [1.63306633e-03 1.33565368e-04 9.30064380e-04 1.52823344e-03
 1.95712660e-04 1.25500071e-03 3.03281619e-03 1.63580842e-03
 9.36254338e-06 1.02557062e-03]
"""
