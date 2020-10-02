from thesiaNet.tensor import tensor
import numpy as np


class Batch:
    def __init__(self,input, target, batch_size= 32, shuffle = True):
        """
        this class conver data into the batches which given size
        :param input: input of the model
        :param target: original output
        :param batch_size: bach size
        :param shuffle: Boolean values (True : shuffeling)
        """
        self.batch_size = 32
        self.shuffle = shuffle
        self.starts = np.arange(0, len(input), self.batch_size)
        if shuffle:
            np.random.shuffle(self.starts)

        batch = []
        for start in self.starts:
            input_batch = input[start:start + self.batch_size]
            target_batch = target[start:start + self.batch_size]
            batch.append((input_batch, target_batch))

        self.batch = batch



