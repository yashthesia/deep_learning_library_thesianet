import numpy as np

input = np.random.randn(1000, 10)
input = np.argmax(input, axis=-1)

x = np.random.randn(1000, 10)
x = np.argmax(x, axis=-1)
print(np.where(x == input))
