from femtotorch.tensor import Tensor

 # Primitive loss functions using building block functions
def softmax(x):
    """
    Output a well-defined probability distritbution: 
        - entries > 0 because passed through exp 
        - entries sum up to 1 because scaled down by the sum of the exponentiated entries
    """

    shifted = (x - x.max(axis=-1, keepdims=True))
    e = shifted.exp()
    return e / (e.sum(axis = -1, keepdims=True)) # 
 
def cross_entropy(x: Tensor, target):
    """
    Harcoded cross_entropy function to compose with softmax on classification problem with one target
    The Tensor x is a batch of vectors of proba (batch_size * 10)
    The Tensor target is a batch of one-hot vectors (batch_size * 10) so it acts as a mask : zeroing the non targeted entries
    """

    out = x * target # entry wise * so every element is 0 except the one hot 
    out_scalar = out.max(axis = -1) # flatten every vector to a scalar which is the proba of the correct digit
    return -(out_scalar.log()) # Outputs negative log likelihood loss tensor


# Optimized loss functions
def softmax_cross_entropy(x: Tensor, target):
    """
    Created because of numerical unstability which started with vgg_batchnorm with 3 conv layers (starts to have quite a few layers),
    And to avoid taking the log of underflowed 0 probability which is undefined (-inf)
    """
    return x.softmax_cross_entropy(target)