from femtotorch.backend import xp
from contextlib import contextmanager
from femtotorch import engine
from femtotorch import operations as ops



class no_grad():
    """
    Implementing __enter__ and __exit__ makes of this class a context manager.
    It disables autograd inside its block, by flipping Tensor.precisison class' variable

    with no_grad():
        preds = model(x)      # inference mode no grad graph is build, no backpropagation

    
    """
    def __enter__(self):
        self.prev = Tensor.grad_mode  # save original state
        Tensor.grad_mode = False
        return self    
    def __exit__(self, *exceptions):
        Tensor.grad_mode = self.prev


def _forward(function: ops.Function, *inputs, **params):
        """
        Creates a new 'out' Tensor result of the function's inputs Tensors,
        while saving in a Node the function and inputs' infos for backward

        This is the only place grad_mode is checked in the entire codebase
        The single seam between Tensor-land and array-land —
    and the only grad_mode check in the entire codebase."""
        context = engine.Node(function, inputs)
        out = Tensor(function.forward(context, *[t.data for t in inputs], **params))

        if Tensor.grad_mode:
            out.grad_node = context # reference Tensor -> Node that allows saving infos for backprobation

        # else: the context Node is not attached to Tensor grad_node attribute 
        #       so,refcounting frees 'context' and it's imediatly garbage collected

        return out

def _as_tensor(x):
    """
    Mainly used to convert scalar into a Tensor.
    """
    return x if isinstance(x, Tensor) else Tensor(x)



class Tensor:
    """
    A tensor is a wrapper that retains a cluster of weights: self.data
    The weights' gradients to be able to update them: self.grad
    The backward rule to backpass the gradients to the Tensors that created

    """
    grad_mode = True # class variable to toggle on and off for a complete 
    precision = xp.float32


    def __init__(self, data, dtype = None):
        
        if dtype is None:
            # takes the precision of data (that comes from previous Tensors' data)
            dtype = getattr(data, "dtype", Tensor.precision) # or fallback to Tensor.precision

        self.data = xp.asarray(data, dtype = dtype)

        self.grad = None # updated with _grafbackward()
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


    
    def __add__(self, other):
        return _forward(ops.Add, self, _as_tensor(other))
    

    
    
    def __mul__(self, other):

    def __pow__(self, other):

    def relu(self):

    def exp(self):

    def log(self):

    def sum(self, axis = None, keepdims=False):
        return _forward(ops.Sum, self, axis = axis, keepdims = keepdims)
    
    def max(self, axis=None, keepdims=False):
    
    def mean(self, axis=None, keepdims=False):

    def __matmul__(self, other):

    def __getitem__(self, key): # self[key] index accessing operation

    def stack(tensors): # tensors is a list of tensor typically a list of result of convolutions in CNN

    def pad_zeros(self, *pad_width):

    def reshape(self, *shape): #*shape so it pack input arguement in a tupple

    def swapaxes(self, axis1, axis2):

    def im2col(self, kernel_size, stride): 

    def softmax_cross_entropy(self, target: "Tensor"):

if __name__ == "__main__":
    a = Tensor(xp.array([2]))
    b = Tensor(xp.array([1]))
    c = a + b
    c.backward()
    print(f"c:{c}, a:{a}, b:{b}")