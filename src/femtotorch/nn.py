import numpy as np
from femtotorch.loss import crossEntropy_MNIST, softmax
from  femtotorch.tensor import Tensor
# Initialize the random number generator
rng = np.random.default_rng()



class Layer():

    def __init__(self, nin, nout, activation = True):
        self.W = Tensor(rng.uniform(-1.0, 1.0, size= (nin, nout)))
        self.B = Tensor(np.zeros((1, nout)))
        self.activation = activation

    def __call__(self, X): # forward pass but with Layer(X)
        linear = (X @ self.W) + self.B # the bias vector self.B is broadcasted
        return linear.relu() if self.activation else linear
    
    def parameters(self):
        return [self.W, self.B] # python list returns W and B reference
    
    def zero_grad(self):
        self.W.zero_grad()
        self.B.zero_grad()

class MLP():

    def __init__(self, nin, nouts):

        sizes = [nin] + nouts # each contiguous couple of integers in this list gives info for layers
        self.layers = []

        # Iterate through to create all hidden layers with ReLU
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=True))
            
        # Append the final output layer without ReLU activation
        self.layers.append(Layer(sizes[-2], sizes[-1], activation=False))

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def parameters(self):
        # use list comprehension syntax to flatten all sublists [self.W, self.B] in an unique list without sublist
        return [p for layer in self.layers for p in layer.parameters()]