import numpy as np
from femtotorch.tensor import Tensor

class Dataloader:
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.X) # len on np array gives size of most outer dimension
        for left_point in range(0, n, self.batch_size): # iterate from 0 to number of elements with step of size batch
            right_point = left_point + self.batch_size
            yield Tensor(self.X[left_point:right_point]), self.Y[left_point:right_point] # pause without stopping the loop