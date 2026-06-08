import numpy as np

class Tensor:
    """
    The fundamental build block of deep learning.

    A Tensor is an n-dimensional array storing real numbers (typically 32-bit floats)
    Using this elementary brick, we can describe all the abstractions we need:
    
    - Architecture: The model is defined by a sequence of operations on weight tensors
    - Signal: The data flowing forward through the model is represented by successive tensors of intermediate values
    - Gradients: The variation dependencies between each layer are captured by gradient tensors
    """
    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=np.float32) # the prefix "as"array avoid copy when data is already well formated
        self.grad = np.zeros_like(self.data) # array of 0's with the same format as data
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
            self.grad += out.grad # use __iadd__ numpy implementation because numpy arrays are mutable
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.multiply(self.data, other.data), (self, other))

        def _backward():
            self.grad += np.multiply(other.data, out.grad)
            other.grad += np.multiply(self.data, out.grad)

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "does not support Tensor^Tensor only int/float powers"

        out = Tensor(np.power(self.data, other), (self,))
        def _backward():
            self.grad += other * np.power(self.data, other - 1) * out.grad # d(out)/dself = d(self^n)/dself = n * self ^(n-1)
            # other is not a variable so need to have its gradient
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
            self.grad += out.grad # taking advantage of numpy broadcasting since out.grad is usually smaller
        out._backward = _backward
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.matmul(self.data, other.data), (self, other))
        def _backward():
            self.grad += out.grad @ np.swapaxes(other.data, -2, -1) # use numpy built in __matmul__ 
            other.grad += np.swapaxes(self.data, -2, -1) @ out.grad
        out._backward = _backward
        return out
    


    def __getitem__(self, key): # self[key] index accessing operation
        out = Tensor(self.data[key], (self,)) # using numpy __getitem__
        
        def _backward():
            np.add.at(self.grad, key, out.grad)   # handles repeated indices safely to avoid overwriting
            self.grad += grad
        out._backward = _backward
        return out
    




    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
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
    

    # to reset gradient after each gradient descent
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    # Construction of the neural net graph and gradient descent
    def backward(self):
        # topological order of all the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data) # array of 1's the same shape as data
        for v in reversed(topo):
            v._backward()
    
    


    # property makes the method behaves like an attribute, so can call .shape .size .ndim
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def ndim(self): # dimensions's number
        return self.data.ndim

if __name__ == "__main__":
    print("test area:")
    
