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

    # def matmul(self, other):
        


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
    a = Tensor([2,3])
    c = a ** 2
    c.backward()
    print(a.grad)   # d(x^2)/dx = 2x = [4., 6.]
