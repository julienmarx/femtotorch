import numpy as np
from femtotorch.engine_switch import Tensor, no_grad

class Dataloader:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle # Add a flag to toggle shuffling

    def __iter__(self):
        n = len(self.X) # len on np array gives size of most outer dimension

        # Create an array of indices [0, 1, 2, ..., n-1]
        indices = np.arange(n)
        
        # Shuffle the indices at the beginning of each iteration
        if self.shuffle:
            np.random.shuffle(indices)


        for left_point in range(0, n, self.batch_size): # iterate from 0 to number of elements with step of size batch
            right_point = left_point + self.batch_size

            # slice 'batch_size' indices in the now randomized array of indices
            batch_indices = indices[left_point:right_point] # NumPy slice that extends beyond the end is silently truncated, useful for last batch
            
            # take advantage of numpy indexing
            yield Tensor(self.X[batch_indices]), self.Y[batch_indices] # pause without stopping the loop