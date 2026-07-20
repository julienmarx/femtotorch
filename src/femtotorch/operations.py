from femtotorch.backend import xp, sliding_window_view
from femtotorch.engine import unbroadcast, broadcast_back, Node


class Function():
    """
    'Abstract' class to enforce forward and backward on subclasses.
    Each subclass is a primitive and takes raw arrays 

    forward(node, *arrays, **params) -> array

    backward(node, g) -> tuple, one grad per Tensor input to be able to backpass the grad

    """

    
    def forward(node, *arrays, **params):
        """Compute output"""
        raise NotImplementedError

    
    def backward(node, g):
        """Push node's gradient to its original Tensor inputs"""
        raise NotImplementedError

class Add(Function):
    @staticmethod
    def forward(node: Node, a, b):
        """
        Since d(a+b)/da = d(a+b)/db = 1, for the backpass,
        we just need shapes to be able to unbroadcast if necessary.
        """
    
        same_shape = a.shape == b.shape
        node.save(None if same_shape else a.shape,
                   None if same_shape else b.shape)

        return xp.add(a, b)
    
    @staticmethod
    def backward(node: Node, g):
        """
        Takes the saved context of Node and the gradient that has been accumulated in the current Tensor,
        and computes the gradients that should be backpassed to Node's inputs
        """
        a_shape, b_shape = node.saved
        return (unbroadcast(g, a_shape), unbroadcast(g, b_shape)) # if None because same_shape, pass through unbroadcast
        
    

class Sum(Function):
    @staticmethod
    def forward(node: Node, x, axis=None, keepdims=False):
        """
        Let e be any scalar entry of x, for any x.sum(axis,keepdim), d(sum(d, e, f))/de = 1,
        so we dont need to store x.data 
        """
        node.save(x.shape, axis, keepdims)

        return xp.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(node: Node, g):
        """
        d(sum(x, axis = None))/d(x[0]) = 1, but it'll just pass g to the correct entries
        Since sum is a reduction function, use broadcast_back helper to reshape
        to the original input's shape
        """
        x_shape, axis, keepdims = node.saved
        return (broadcast_back(g, x_shape, axis, keepdims),) # returns a tuple

class Mul(Function):
    @staticmethod
    def forward(node: Node, a, b):
        """
        d(a*b)/da = b and d(a*b)/db = a, so we need to save the whole 2 inputs
        """
        node.save(a, b)

        return xp.multiply(a, b)
    
    @staticmethod
    def backward(node: Node, g):
        """
        returns (a.grad = g * b, b.grad = g * a)
        """
        a, b = node.saved

        return (unbroadcast(xp.multiply(b, g), a.shape), unbroadcast(xp.multiply(a, g), b.shape))

class Pow(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Relu(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Exp(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Log(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Max(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Mean(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Matmul(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Getitem(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Stack(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class PadZeros(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Reshape(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Swapaxes(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class Im2col(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

class SoftMaxCrossEntropy(Function):
    @staticmethod
    def forward(node: Node,):
    @staticmethod
    def backward(node: Node, g):

            



        

    
