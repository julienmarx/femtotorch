import numpy as np
from  femtotorch.tensor import Tensor
# Fused loss function will have to be implemented to avoid taking log of proba which can be if unlucky 0 so undefined log


 # loss functions using building block functions (essentially to check 


def softmax(self):
    shifted = (self - self.max(axis=-1, keepdims=True))
    e = shifted.exp()
    return e / (e.sum(axis = -1, keepdims=True))

#the tensor self is a batch of vectors of proba (batch_size * 10)
#the tensor target is a batch of one-hot vectors (batch_size * 10) so it acts as a mask
def crossEntropy_MNIST(self: Tensor, target):
    out = self * target # entry wise * so every element is 0 except the one hot 
    out_scalar = out.max(axis = -1) # flatten every vector to a scalar which is the proba of the correct digit
    return -(out_scalar.log())

