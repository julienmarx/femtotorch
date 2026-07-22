from femtotorch.backend import xp
from contextlib import contextmanager
from femtotorch import engine
from femtotorch import operations as ops



class no_grad():
    """
    Implementing __enter__ and __exit__ makes of this class a context manager.
    It disables autograd inside its block, by flipping Tensor.grad_mode class' variable

    with no_grad():
        preds = model(x)      # inference mode no grad graph is build, no backpropagation

    
    """
    def __enter__(self):
        self.prev = Tensor.grad_mode  # save original state
        Tensor.grad_mode = False
        return self
     
    def __exit__(self, *exceptions):
        Tensor.grad_mode = self.prev


def _forward(function: ops.Function, *inputs, **params): # ops.Function is an instance of Type
        """
        Creates a new 'out' Tensor result of the function's inputs Tensors,
        while saving in a Node the function and inputs' infos for backward

        This is the only place grad_mode is checked in the entire codebase
        """
        context = engine.Node(function, inputs)
        out = Tensor(function.forward(context, *[t.data for t in inputs], **params))

        if Tensor.grad_mode:
            out.grad_node = context # reference Tensor -> Node that allows saving infos for backpropagation

        # else: the context Node is not attached to Tensor grad_node attribute 
        #       so,refcounting frees 'context' and it's imediatly garbage collected

        return out

def _as_tensor(x, dtype = None):
    """
    Mainly used to convert scalar into a Tensor.
    """
    return x if isinstance(x, Tensor) else Tensor(x, dtype=dtype)



class Tensor:
    """
    A tensor is a wrapper that retains a cluster of weights: self.data
    The weights' gradients to be able to update them: self.grad
    The backward rule to backpass the gradients to the Tensors that created

    """
    grad_mode = True # class variable to toggle on/off for training / inference 
    precision = xp.float32


    def __init__(self, data, dtype = None):
        
        if dtype is None:
            # takes the precision of data (that comes from previous Tensors' data)
            dtype = getattr(data, "dtype", Tensor.precision) # or fallback to Tensor.precision

        self.data = xp.asarray(data, dtype = dtype)

        self.grad = None # updated with backward()
        self.grad_node = None # updated with _forward() (for a leaf it'll always be None)

    def zero_grad(self):
        self.grad = None
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(data ={self.data}, grad= {self.grad})"


    def backward(self):
        """
        Backpropagates gradients with self as graph's root.
        """
        engine.graph_backward(self)
    

    def _accumulate_grad(self, g):
        """
        Avoid unnecessary copies by just passing the reference if there is no accumulation of different gradients: self.grad = g
        """
        assert g.shape == self.data.shape, f"op returned grad {g.shape} for tensor {self.data.shape} — missing unbroadcast?"
        assert g.dtype == self.data.dtype, f"op leaked dtype {g.dtype} into {self.data.dtype} tensor"

        if self.grad is None: # no initialize yet (because of lazy initialization)
            self.grad = g # just copy the reference
        else: # already initialized
            self.grad = g + self.grad # allocates a new object for self.grad to avoid mutating gradients of views operations


    # for inference only:

    def argmax(self, axis=None, keepdims=False):
        """
        Used to get the index of the max value (in other words the predicted class)
        """
        return Tensor(xp.argmax(self.data, axis=axis, keepdims=keepdims))
    

    # for training and inference:

    def __add__(self, other):
        return _forward(ops.Add, self, _as_tensor(other, dtype = self.data.dtype)) # precise the dtype to avoid dtype leak if other is a scalar
    
    def __mul__(self, other):
        return _forward(ops.Mul, self, _as_tensor(other, dtype = self.data.dtype))

    def __pow__(self, other):
        assert isinstance(other, (float, int)), "does not support Tensor^Tensor only int/float powers"
        return _forward(ops.Pow, self, exponent = other) # other is a number

    def relu(self):
        return _forward(ops.Relu, self)
    
    def exp(self):
        return _forward(ops.Exp, self)

    def log(self):
        return _forward(ops.Log, self)

    def sum(self, axis = None, keepdims=False):
        return _forward(ops.Sum, self, axis = axis, keepdims = keepdims)
    
    def max(self, axis=None, keepdims=False):
        return _forward(ops.Max, self, axis = axis, keepdims = keepdims)
    
    def mean(self, axis=None, keepdims=False):
        return _forward(ops.Mean, self, axis = axis, keepdims = keepdims)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else _as_tensor(other)
        assert self.data.ndim >= 2 and other.data.ndim >= 2, "matmul expects ≥2D operands: a single example is (1, D), not (D,)"
        return _forward(ops.Matmul, self, other)

    def __getitem__(self, key): # self[key] index accessing operation
        return _forward(ops.Getitem, self, key=key)
    
    @staticmethod
    def stack(tensors): # tensors is a list of tensor typically a list of result of convolutions in CNN
        # * tensors to unpack the list' elements so _forward catch them and store them in the *input tuple
        return _forward(ops.Stack, *tensors) 

    def pad_zeros(self, *pad_width): #pad_width is a tuple of tuples (giving the number of zeros in front and behind each axis)
        return _forward(ops.PadZeros, self, pad_width = pad_width)

    def reshape(self, *shape): # shape is a tuple
        return _forward(ops.Reshape, self, shape = shape)

    def swapaxes(self, axis1, axis2):
        return _forward(ops.Swapaxes, self, axis1 = axis1, axis2 = axis2)

    def im2col(self, kernel_size, stride): 
        return _forward(ops.Im2col, self, kernel_size = kernel_size, stride = stride)

    def softmax_cross_entropy(self, target): # target is a Tensor of indices of class
        t = target.data if isinstance(target, Tensor) else xp.asarray(target)
        return _forward(ops.SoftMaxCrossEntropy, self, target = t)

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