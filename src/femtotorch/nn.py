import numpy as np
from  femtotorch.tensor import Tensor
# Initialize the random number generator
rng = np.random.default_rng()



class Layer():

    def __init__(self, nin, nout):
        self.W = Tensor(rng.uniform(-1.0, 1.0, size= (nin, nout)))
        self.B = Tensor(np.zeros(1, nout))
    
    def __call__(self, X): # forward pass but with Layer(X)
        linear = (X @ self.W) + self.B
        return linear.
        
        