from femtotorch.backend import xp as np, sliding_window_view # for im2col

from contextlib import contextmanager

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

    grad_mode = True

    def __init__(self, data, _prev=(), dtype = None):
        
        if dtype is None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
            else:
                dtype = np.float32

        self.data = np.asarray(data, dtype = dtype) # the prefix "asarray" avoids copying when data is already well formatted
        # array of zeros with the same shape and dtype as data
        self.grad = None # Lazy evaluation
        self._prev = set(_prev) if Tensor.grad_mode else set()
        self._backward = lambda: None # by default a dummy function returning None
        # for leaf nodes in the autograd engine

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.add(self.data, other.data), (self, other))

        if not Tensor.grad_mode: 
            return out # without the closure below self and other are garbage collected

        def _backward():
            self._accumulate_grad(unbroadcast(out.grad, self.shape))   # accumulate contributions from each out node depending self node
            other._accumulate_grad(unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(np.multiply(self.data, other.data), (self, other))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(unbroadcast(np.multiply(other.data,out.grad), self.shape))
            other._accumulate_grad(unbroadcast(np.multiply(self.data, out.grad), other.shape))

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "does not support Tensor^Tensor only int/float powers"

        out = Tensor(np.power(self.data, other), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(other * np.power(self.data, other - 1) * out.grad) # d(out)/dself = d(self^n)/dself = n * self ^(n-1)
            # other is not a variable and does not receive gradients
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,)) # np.maximum is the entry wise version of np.max

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad((out.data > 0) * out.grad)
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(out.data * out.grad)
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(out.grad / self.data)
        out._backward = _backward
        return out
    
    def sum(self, axis = None, keepdims=False):
        out = Tensor(np.sum(self.data, axis = axis, keepdims = keepdims), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            # handle cleanly the broadcasting according to the collapsed axis during forward pass
            grad = out.grad # to keep out.grad in its own shape
            if not keepdims and axis is not None: # if axis is None sum flatten everything into an scalar which the default broadcasting handle well
                grad = np.expand_dims(grad, axis) # reshape out.grad to make is match self.grad

            self._accumulate_grad(grad)
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,))

        if not Tensor.grad_mode:
            return out

        def _backward():
            grad = out.grad
            max_vals = out.data
             
            if not keepdims and axis is not None: # cases where grad/data needs explicit broadcasting to fit correctly self.grad
                grad = np.expand_dims(grad, axis)
                max_vals = np.expand_dims(max_vals, axis)

            mask = (self.data == max_vals) # makes a mask with ones for each index of the elements = max
            counts = np.sum(mask, axis=axis, keepdims=True) # number of element equals to max 
            self._accumulate_grad(mask * grad / counts) # share the grad to each contributing element
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis = axis, keepdims=keepdims)

        if axis is None: # in other words, f(self: tensor) = scalar
            n = self.size

        elif isinstance(axis, int): # a single selected axis
            n = self.shape[axis] 

        else:
            n = 1
            for a in axis: # axis is a tuple (so a least 2 selected axis)
                n *= self.shape[a]

        return s * 1 / n


    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # matmul contract to avoid shape managing
        assert self.data.ndim >= 2 and other.data.ndim >= 2, "matmul expects ≥2D operands: a single example is (1, D), not (D,)"

        out = Tensor(np.matmul(self.data, other.data), (self, other))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(out.grad @ np.swapaxes(other.data, -2, -1)) # use numpy built in __matmul__
            other._accumulate_grad(np.swapaxes(self.data, -2, -1) @ out.grad)
        out._backward = _backward
        return out
    


    def __getitem__(self, key): # self[key] index accessing operation
        out = Tensor(self.data[key], (self,)) # using numpy __getitem__
        
        if not Tensor.grad_mode:
            return out
        
        def _backward():
            grad = np.zeros_like(self.data)
            # handles repeated indices safely,
            # If there are repeated indices, it adds each out.grad to the corresponding self[key]
            np.add.at(grad, key, out.grad)
            self._accumulate_grad(grad)
        out._backward = _backward
        return out
    
    @staticmethod # static because we dont want to consider one of the tensor self and create arbitrary hierarchy
    def stack(tensors): # tensors is a list of tensor typically a list of result of convolutions in CNN
        # only support stack on axis = 0
        out = Tensor(np.stack([t.data for t in tensors]), set(tensors))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            # out.data has shape (N, *t.shape); slice i is the i-th input's grad, the i-th tensor
            # concretly out.data is a column vector of tensors
            for i, t in enumerate(tensors):
                t._accumulate_grad(out.grad[i])
        out._backward = _backward

        return out
    
    def pad_zeros(self, *pad_width):
        out = Tensor(np.pad(self.data, pad_width = pad_width, mode='constant', constant_values=0), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            crop = []
            for axis, (front_pad, back_pad) in enumerate(pad_width):
                out_length = out.data.shape[axis]
                crop.append(slice(front_pad, out_length - back_pad)) # slice is [front_pad, out_length - back_pad]

            self._accumulate_grad(out.grad[tuple(crop)]) # use [((frond pad, out_length-back_pad), ...)]
        
        out._backward = _backward

        return out

    def reshape(self, *shape): #*shape so it pack input arguement in a tupple
        out = Tensor(self.data.reshape(shape), (self,)) # (self,) not self: _prev must be a tuple of parents, else set() iterates the Tensor

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward

        return out

    def swapaxes(self, axis1, axis2):
        out = Tensor(self.data.swapaxes(axis1, axis2), (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            self._accumulate_grad(out.grad.swapaxes(axis1, axis2))

        out._backward = _backward
        return out

    def im2col(self, kernel_size, stride): 
        """
        self: Tensor of shape (batch, inchanel, height, width) expected to already be padded

        kernel_size: considering a square kernel size
        """
        batch, in_channel, height, width = self.data.shape
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1 

        # if float32 each scalar in 4 bytes
        # address(b, in_c, h, w) = base + w*4 + h*(w*4) + in_c * (h*w*4)

        # returns a new metadata header so shape is (b, in_c, h-(k-1), w-(k-1), k, k)
        # the map from header to memory becomes non injective (multiple elements of "window" corresponds to the same memory addres)
        # strides = ( in_c * (h*w*4), h*w*4, 4*width, 4, 4*width, 4)
        window = sliding_window_view(self.data, (kernel_size, kernel_size), axis=(-2, -1))
        window = window[:, :, ::stride, ::stride, :, :]     # still shape 6D but (b, in_c, (h-(k-1) // stride)+1, (w-(k-1) // stride)+1, k, k) 
        window = window.transpose(0, 2, 3, 1, 4, 5)   # shape (batch, out_height, out_width, in_c, k, k)
        # reshape is the first operation that doesnt just change the view but copy the data so it's contiguous
        cols = window.reshape(-1, in_channel * kernel_size * kernel_size) # finally the real physical gather
        
        out = Tensor(cols, (self,))

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            grad = out.grad.reshape(-1, out_height, out_width, in_channel, kernel_size, kernel_size)
            grad = grad.transpose(0, 3, 1, 2, 4, 5)
            
            grad_to_add = np.zeros(self.data.shape, dtype=out.grad.dtype)

            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    grad_to_add[:, :,
                                #  start = ky, step = stride, stop = ky + stride*out_h
                                ky : ky + stride * out_height: stride, # so it selects out_height rows
                                # it selects out_width columns, so grad_to_add have shape
                                kx : kx + stride * out_width : stride] += grad[:, :, :, :, ky, kx] # axis 0 and 1 are selected so shape (batch, in_channel, out_height, out_width)
            
            self._accumulate_grad(grad_to_add) # shape (batch, inchanel, height, width)

        out._backward = _backward

        return out # shape (batch * out_width * out_height, fan_in)
    
    def softmax_cross_entropy(self, target: "Tensor"):
        """
        Fused sofmax and cross entropy to prevent division by 0 in the backward because of underflow
        """
        # target: (batch_size,) int indices
        target = np.asarray(target.data, dtype = int)
        rows = np.arange(len(target)) # [0, 1, ..., batch_size-1]

        max = self.data.max(axis=-1, keepdims = True) # (batch_size,1)
        e = np.exp(self.data - max)  # (batch_size, num_classes)

        sum_e = e.sum(axis = -1, keepdims=True)   # (batch_size,1)

        out = Tensor(np.log(sum_e).squeeze(-1) + max.squeeze(-1) - self.data[rows, target], (self,)) 
        

        if not Tensor.grad_mode:
            return out
        
        def _backward():
            grad = e / sum_e
            grad[rows, target] -= 1.0
            self._accumulate_grad(grad * out.grad[:,np.newaxis])

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

            # reversed(topo) runs consumers before their inputs (nodes in the _prev of consumers),
            #  so by the time v._backward() has run, v.grad has received every contribution and nothing will read it again



            # release the graph: break the circular references out -> _backward -> out
            # this makes a second call loss.backward() a no-op since the graph does not exist anymore

            if v._prev: # if v._prev is non-empty, in other words v is an intermediat node (not a leaf)
                v.grad = None
            
            # Restoring constructor state, lambda: None and not just 'None' to be able build_topo on the second batch
            v._backward = lambda : None # severs closure
            v._prev.clear() # empties the set of parent references

            

    # for inference
    def argmax(self, axis=None, keepdims=False):
        out = Tensor(np.argmax(self.data, axis=axis, keepdims=keepdims), (self,))
        return out


    # getter functions
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def ndim(self): # number of dimensions
        return self.data.ndim

   
    def _accumulate_grad(self, g):
        if self.grad is None:
            self.grad = np.broadcast_to(g, self.data.shape).astype(self.data.dtype)
        else:
            self.grad += g






@contextmanager #allows to use "with statements"
def no_grad():
    Tensor.grad_mode = False

    try:
        yield   # the body of the "with" block runs

    finally:
        Tensor.grad_mode = True # always restored, even if the "with" block crashes
    
  



    


