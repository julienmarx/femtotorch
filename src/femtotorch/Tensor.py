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

        out = Tensor(np.multiply(self, other))

        def _backward():
            self.grad += np.multiply(other.grad * out.grad)
            other.grad += np.multiply(self.grad * out.grad)

    
    

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
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"




    def shape(self):
        return self.data.shape
    
    def size(self):
        return self.data.size

    def ndim(self): # dimensions's number
        return self.data.ndim

if __name__ == "__main__":
    print("Let's go !")
    a = Tensor(np.array([1,1,1]))
    b = Tensor(np.array([1,1,1]))
    c = a + b
    print(a, b, c)
    print("backward")
    c.backward()
    print(a, b, c)
