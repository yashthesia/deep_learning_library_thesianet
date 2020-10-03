#this file contains some of the visualization of the weight intialization


import numpy as np



import matplotlib.pyplot as plt

fig,a =  plt.subplots(2,5)

X = np.random.rand(1000,500)
weights = np.random.rand(10,500,500)/np.sqrt(250)

Distribution = [X]
input = X

for i in range(weights.shape[0]):
    print(int(i/5),i%5)
    a[int(i/5)][i%5].hist(input, bins=50)
    input = np.tanh(X @ weights[i])
    Distribution.append(input)


plt.show()


mean = [[x.mean()] for x in Distribution]
var =  [[x.var()] for x in Distribution]
plt.plot(mean)
plt.title("means shift")
plt.show()

plt.plot(var)
plt.title("varience shift")
plt.show()



