#this file contains some of the visualization of the weight intialization
import numpy as np
import matplotlib.pyplot as plt


def forward(activation_name, input):
    """
    :param input: data from the direct input or previous layer
    :return:  forward pass of the Liner layer
    """
    print(activation_name)

    if activation_name == "tanh":
        print("tanh")
        out = np.tanh(input)
        return out

    if activation_name == "relu":
        print("relu")
        out = np.maximum(input, 0)
        return out

    if activation_name == "sigmoid":
        print("sigmoid")
        out = 1 / (1 + np.exp(input * -1))
        return out

    if activation_name == "linear":
        out = input
        return out

    if activation_name == "softmax":
        print("softmax")
        out = np.exp(input)
        out = out / (out.sum(axis=0) + 1e-10)
        return out


features = 512
X = np.random.normal(size=(1000,features))
weights = [512]*10
activation = ['tanh']*10
w = np.random.randn(10,512,512)# intialization of weights

fig, a = plt.subplots(2, 5)

input = X
Distribution = [X]


for i in range(len(w)):
    print(int(i / 5), i % 5)
    a[int(i / 5)][i % 5].hist(input.ravel(),30,range = (-1,1))

    input = forward(activation[i], np.dot(input,w[i]))
    print(f"mean : {input.mean()} | std : {input.std()}")
    Distribution.append(input)

plt.show()

mean = [[x.mean()] for x in Distribution]
var =  [[x.std()] for x in Distribution]
plt.plot(mean)
plt.title("means shift")
plt.show()

plt.plot(var)
plt.title("varience shift")
plt.show()