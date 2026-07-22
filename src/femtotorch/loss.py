from femtotorch.tensor import Tensor
# Fused loss function will have to be implemented to avoid taking log of proba which can be if unlucky 0 so undefined log


 # loss functions using building block functions (essentially to check 


def softmax(x):
    shifted = (x - x.max(axis=-1, keepdims=True))
    e = shifted.exp()
    return e / (e.sum(axis = -1, keepdims=True))

#the tensor x is a batch of vectors of proba (batch_size * 10)
#the tensor target is a batch of one-hot vectors (batch_size * 10) so it acts as a mask
def cross_entropy(x: Tensor, target):
    out = x * target # entry wise * so every element is 0 except the one hot 
    out_scalar = out.max(axis = -1) # flatten every vector to a scalar which is the proba of the correct digit
    return -(out_scalar.log())

def softmax_cross_entropy(x: Tensor, target):
    """
    Created because of numerical unstability which started with vgg_batchnorm with 3 conv layers (starts to have quite a few layers)
    """
    return x.softmax_cross_entropy(target)