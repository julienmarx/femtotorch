import numpy as np

# helper function to allow backward pass on operations with numpy broadcasting
def unbroadcast(outGrad, shape):
    # handle the case of an operation between arrays of different dimensional space
    # typically a bias vector (n,)
    while outGrad.ndim > len(shape):
        outGrad = outGrad.sum(axis=0)
    # handle the case of different dimensional object in the same space
    # typically a bias vector (1, n)
    for i, dim in enumerate(shape):
        if dim == 1:
            outGrad = outGrad.sum(axis=i, keepdims=True)

    return outGrad # Returns the unbroadcasted gradient array to use in the chain rule

class Tensor:
    """
    The fundamental building block of deep learning.

    A Tensor is an n-dimensional array storing real numbers (typically 32-bit floats)
    Using this elementary brick, we can describe all the abstractions we need:
    
    - Architecture: The model is defined by a sequence of operations on weight tensors
    - Signal: The data flowing forward through the model is represented by successive tensors of intermediate values
    - Gradients: The variation dependencies between each layer are captured by gradient tensors
    """
    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=np.float32) # the prefix "asarray" avoids copying when data is already well formatted
        self.grad = np.zeros_like(self.data) # array of zeros with the same shape and dtype as data
        self._prev = set(_prev)
        self._backward = lambda: None # by default a dummy function returning None
        # for leaf nodes in the autograd engine

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.add(self.data, other.data), (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.shape()) # use += to accumulate contributions from each out node depending self node
            other.grad += unbroadcast(out.grad, other.shape())

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.multiply(self.data, other.data), (self, other))

        def _backward():
            self.grad += unbroadcast(np.multiply(other.data,out.grad), self.shape())
            other.grad += unbroadcast(np.multiply(self.data, out.grad), other.shape()) 

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "does not support Tensor^Tensor only int/float powers"

        out = Tensor(np.power(self.data, other), (self,))
        def _backward():
            self.grad += other * np.power(self.data, other - 1) * out.grad # d(out)/dself = d(self^n)/dself = n * self ^(n-1)
            # other is not a variable and does not receive gradients
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,)) # np.maximum is the entry wise version of np.max
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,))
        def _backward():
            self.grad += out.grad / self.data 
        out._backward = _backward
        return out
    
    def sum(self, axis = None, keepdims=False):
        out = Tensor(np.sum(self.data, axis = axis, keepdims = keepdims), (self,))
        def _backward():
            # handle cleanly the broadcasting according to the collapsed axis during forward pass
            grad = out.grad # to keep out.grad in its own shape
            if not keepdims and axis is not None: # if axis is None sum flatten everything into an scalar which the default broadcasting handle well
                grad = np.expand_dims(grad, axis) # reshape out.grad to make is match self.grad

            self.grad += grad
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,))

        def _backward():
            grad = out.grad
            # keepdims to have a mask which has self.grad shape
            input_shape = np.max(self.data, axis=axis, keepdims=True)
            mask = (self.data == input_shape) # makes a mask with ones for each index of the elements = max

            if not keepdims and axis is not None: # cases where grad needs explicit broadcasting to fit correctly self.grad
                grad = np.expand_dims(grad, axis)
            counts = np.sum(mask, axis=axis, keepdims=True) # number of element equals to max 

            self.grad += mask * grad / counts # share the grad to each contributing element
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis = axis, keepdims=keepdims)
        n = self.size() if axis is None else self.size()[axis] # if axis is None the whole tensor is collapsed into a scalar so n is .size()
        return s * (1.0 / n)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # matmul contract to avoid shape managing
        assert self.data.ndim >= 2 and other.data.ndim >= 2, "matmul expects ≥2D operands: a single example is (1, D), not (D,)"

        out = Tensor(np.matmul(self.data, other.data), (self, other))
        def _backward():
            self.grad += out.grad @ np.swapaxes(other.data, -2, -1) # use numpy built in __matmul__ 
            other.grad += np.swapaxes(self.data, -2, -1) @ out.grad
        out._backward = _backward
        return out
    


    def __getitem__(self, key): # self[key] index accessing operation
        out = Tensor(self.data[key], (self,)) # using numpy __getitem__
        
        def _backward():
            grad = np.zeros_like(self.data)
            # handles repeated indices safely,
            # If there are repeated indices, it adds each out.grad to the corresponding self[key]
            np.add.at(grad, key, out.grad)
            self.grad += grad
        out._backward = _backward
        return out
    

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    # reflected operations to handle NotATensor.__operation__(Tensor) by calling Tensor.__operation__(NotATensor)
    def __rsub__(self, other):
        return (-self) + other
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self ** - 1) * other
    

    # to reset gradient after each gradient descent step
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    # Construction of the computation graph and gradient descent
    def backward(self):
        # Build topological ordering of all nodes in the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data) # array of ones with the same shape as data
        for v in reversed(topo):
            v._backward()

    # for inference
    def argmax(self, axis=None, keepdims=False):
        out = Tensor(np.argmax(self.data, axis=axis, keepdims=keepdims), (self,))
        return out


    # getter functions
    
    def shape(self):
        return self.data.shape
    
    
    def size(self):
        return self.data.size
    
    
    def ndim(self): # number of dimensions
        return self.data.ndim

   
    
  



    


    
    

    
