from thesiaNet.tensor import tensor
from thesiaNet.layers import *
from thesiaNet.nn import *
from thesiaNet.optim import *
from thesiaNet.data import *
from thesiaNet.loss import *
import numpy as np

def train(net = None,
          input = None,
          target = None,
          epochs  = 5000,
          batch_size = 32,
          loss = MSE(),
          optimizer= SGD(),
          epoch_skip = 1):



    for e in range(epochs):
        epoch_loss = 0.0
        batchs = Batch(input, target,batch_size)
        for batch in batchs.batch:
            X, y = batch

            predict = net.forward(X)
            # print(predict.shape)
            epoch_loss += loss.loss(predict, y)

            grad = loss.grad(predict,y)

            net.backward(grad)
            optimizer.step(net)
        if (e+1)%epoch_skip == 0:
            print(f"epoch {e+1} | Loss : {epoch_loss}")




