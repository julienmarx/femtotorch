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
    """
    Element-wise addition
    """

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
        

class Mul(Function):
    """
    Element-wise multiplication (Hadamard product)
    """
    
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
    """
    Element-wise power with a scalar
    """

    @staticmethod
    def forward(node: Node, a, exponent):
        """
        b is always a scalar never a multidimensionnal object
        """
        node.save(a, exponent)

        return xp.power(a, exponent)
    
    @staticmethod
    def backward(node: Node, g):
        """
        d(a^b)/da = b * a^(b-1)
        and we don't adapt b since we always use it as a constant scalar
        """
        a, exponent = node.saved

        return (xp.multiply(exponent, xp.power(a, exponent - 1)) * g,)
        

class Relu(Function):
    """
    Element-wise relu(x)
    """

    @staticmethod
    def forward(node: Node, a):
        """
        returns 0 if element <= 0 or x if elements > 0
        """
        out = xp.maximum(0, a)
        node.save(out)
        return out

    @staticmethod
    def backward(node: Node, g):
        """
        Let x be any entry of the input array,
            max(0, x) = x if x > 0 and 0 if x <= 0
        and since: 
            d(max(0, x))/dx = 1 if x > 0 and 0 if x <= 0 = boolean(max(0, x) > 0)

        So we can just use the data of the output Tensor as local gradient * the incomming gradient
        The output Tensor data is stored in the Node
        """
        out = node.saved[0] # (out,)[0]
        
        return (xp.multiply((out > 0), g),) # xp.greater(out, 0) = out > -


class Exp(Function):
    """
    Element-wise exponentiation.
    """

    @staticmethod
    def forward(node: Node, a):
        exp_a = xp.exp(a)
        node.save(exp_a)
        return exp_a
    
    @staticmethod
    def backward(node: Node, g):
        """
        d(exp(a))da = exp(a)
        """
        exp_a = node.saved[0] # (exp_a,)[0]

        return (xp.multiply(exp_a, g),)


class Log(Function):
    """
    Element-wise log(x), operation to avoid because its derivative is 1/x,
    which can raise RuntimeWarning when x underflows because of 1 / 0
    """

    @staticmethod
    def forward(node: Node, a):
        node.save(a)
        return xp.log(a)
    
    @staticmethod
    def backward(node: Node, g):
        """
        d(log(x))/dx = 1/x
        """
        a = node.saved[0] #(a,)[0]

        return (xp.divide(g, a),)
    

class Sum(Function):
    """
    Sum across axes.
    """

    @staticmethod
    def forward(node: Node, a, axis=None, keepdims=False):
        """
        Let e be any scalar entry of x, for any x.sum(axis,keepdim), d(sum(d, e, f))/de = 1,
        so we dont need to store x.data 
        """
        node.save(a.shape, axis, keepdims)

        return xp.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(node: Node, g):
        """
        d(sum(x, axis = None))/d(x[0]) = 1, but it'll just pass g to the correct entries
        Since sum is a reduction function, use broadcast_back helper to reshape
        to the original input's shape
        """
        x_shape, axis, keepdims = node.saved
        return (broadcast_back(g, x_shape, axis, keepdims),) # returns a tuple


class Max(Function):
    """
    Max across axes.
    """

    @staticmethod
    def forward(node: Node, a, axis=None, keepdims=False):
        out = xp.max(a, axis=axis, keepdims=keepdims)
        mask = (broadcast_back(out, a.shape, axis, keepdims) == a) # makes a mask with ones for each index of the elements = max, with 1-byte dtype
        node.save(mask, axis, keepdims)
        return out
    
    @staticmethod
    def backward(node: Node, g):
        """
        d(max(x, y))/dx = 1 if x is argmax of max(x, y) or 0 if not
        """
        mask, axis, keepdims = node.saved
        # number of element equals to max per out entry, split the grad to each contributing element
        counts = xp.sum(mask, axis=axis, keepdims=True).astype(g.dtype) #counts is int64 so it has to be downcast
        return (xp.multiply(mask, broadcast_back(g, mask.shape, axis, keepdims)) / counts,) # mask.shape == a.shape


class Mean(Function):
    """
    Mean across axes.
    """

    @staticmethod
    def forward(node: Node, a, axis=None, keepdims=False):
        out = xp.mean(a, axis = axis, keepdims = keepdims)
        # the non-reduced axes cancel top-and-bottom, leaving exactly the product of the reduced axes:
        n = a.size // out.size 
        node.save(a.shape, axis, keepdims, n)
        return out
    
    @staticmethod
    def backward(node: Node, g):
        a_shape, axis, keepdims, n = node.saved
        return (xp.multiply(broadcast_back(g, a_shape, axis, keepdims), 1/n),)


class Matmul(Function):
    """
    Matrix multiplication between 2D matrices (with channels if given)
    """

    @staticmethod
    def forward(node: Node, a, b):
        node.save(a, b)
        return xp.matmul(a, b)
    
    @staticmethod
    def backward(node: Node, g):
        a, b = node.saved
        return (xp.matmul(g, xp.swapaxes(b, -1, -2)), xp.matmul(xp.swapaxes(a, -1, -2), g))


class Getitem(Function):
    """
    self[key] index accessing operation
    """

    @staticmethod
    def forward(node: Node, a, key):
        node.save(key, a.shape)
        return a[key]
    
    @staticmethod
    def backward(node: Node, g):
        key, a_shape = node.saved
        grad = xp.zeros(shape = a_shape, dtype = g.dtype)
        # handles repeated indices safely,
        # If there are repeated indices, it adds each out.grad to the corresponding self[key]
        xp.add.at(grad, key, g)
        return (grad,)


class Stack(Function):
    """
    Stack N arrays of shape S into one array of shape (N, *S)
    """

    @staticmethod
    def forward(node: Node, *a): 
        """
        Collects all the raw arrays in the list created by _forward in the tuple 'a'
        """
        return xp.stack(a) # creates the array (len(a), *a[0].shape)
    @staticmethod
    def backward(node: Node, g):
        n = g.shape[0] # number of arrays stacked
        return tuple(g[i] for i in range(n))

class PadZeros(Function):
    @staticmethod
    def forward(node: Node, a, pad_width):
        node.save(pad_width)
        return xp.pad(a, pad_width, mode = 'constant', constant_values = 0)

    @staticmethod
    def backward(node: Node, g):
        (pad_width,) = node.saved
        crop = [] # stores the slice recipe for each axis
        for axis, (front, back) in enumerate(pad_width):
            original_back = g.shape[axis] - back
            crop.append(slice(front, original_back))
        crop = tuple(crop) # tuple of slices object ((axis1_start, axis1_end), (axisn_start, axisn_end)))
        return (g[crop],)


class Reshape(Function):
    @staticmethod
    def forward(node: Node, a, shape):
        node.save(a.shape)
        return a.reshape(shape)
    @staticmethod
    def backward(node: Node, g):
        a_shape = node.saved[0] # (a_shape,)[0]
        return (g.reshape(a_shape),)



class Swapaxes(Function):
    @staticmethod
    def forward(node: Node, a, axis1, axis2):
        node.save(axis1, axis2)
        return a.swapaxes(axis1, axis2)
    @staticmethod
    def backward(node: Node, g):
        axis1, axis2 = node.saved
        return (g.swapaxes(axis1, axis2),)

class Im2col(Function):
    """
    Im2col reshape to have efficient convolutions using matrix multiplications
    """
    @staticmethod
    def forward(node: Node, a, kernel_size, stride):
        """
        a: Tensor of shape (batch, inchanel, height, width) expected to already be padded
        kernel_size: considering a square kernel size
        """
        batch, in_channel, height, width = a.shape
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1 

        node.save(a.shape, kernel_size, stride)

        # if default dtype float32 each scalar in 4 bytes
        # address(b, in_c, h, w) = base + w*4 + h*(w*4) + in_c * (h*w*4)

        # returns a new metadata header so shape is (b, in_c, h-(k-1), w-(k-1), k, k)
        # the map from header to memory becomes non injective (multiple elements of "window" corresponds to the same memory addres)
        # strides = ( in_c * (h*w*4), h*w*4, 4*width, 4, 4*width, 4)
        window = sliding_window_view(a, (kernel_size, kernel_size), axis=(-2, -1))
        window = window[:, :, ::stride, ::stride, :, :]     # still shape 6D but (b, in_c, (h-(k-1) // stride)+1, (w-(k-1) // stride)+1, k, k) 
        window = window.transpose(0, 2, 3, 1, 4, 5)   # shape (batch, out_height, out_width, in_c, k, k)
        # reshape is the first operation that doesnt just change the view but copy the data so it's contiguous
        cols = window.reshape(-1, in_channel * kernel_size * kernel_size) # finally the real physical gather
        
        return cols
    
    @staticmethod
    def backward(node: Node, g):

        a_shape, kernel_size, stride = node.saved

        batch, in_channel, height, width = a_shape
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1 

        grad_to_add = xp.zeros(a_shape, dtype=g.dtype)

        g = g.reshape(-1, out_height, out_width, in_channel, kernel_size, kernel_size)
        g = g.transpose(0, 3, 1, 2, 4, 5)

        for ky in range(kernel_size):
                for kx in range(kernel_size):
                    grad_to_add[:, :,
                                #  start = ky, step = stride, stop = ky + stride*out_h
                                ky : ky + stride * out_height: stride, # so it selects out_height rows
                                # it selects out_width columns, so grad_to_add have shape
                                kx : kx + stride * out_width : stride] += g[:, :, :, :, ky, kx] # axis 0 and 1 are selected so shape (batch, in_channel, out_height, out_width)
        return (grad_to_add,)

class SoftMaxCrossEntropy(Function):
    """
        Fused sofmax and cross entropy to prevent division by 0 in the backward because of underflow
    """
    @staticmethod
    def forward(node: Node, a, target):
        # target: (batch_size,) int indices
        target = target.astype(dtype = xp.int64)
        rows = xp.arange(len(target)) # [0, 1, ..., batch_size-1]

        max = a.max(axis=-1, keepdims = True) # (batch_size,1)
        e = xp.exp(a - max)  # (batch_size, num_classes)

        sum_e = e.sum(axis = -1, keepdims=True)   # (batch_size,1)

        out = xp.log(sum_e).squeeze(-1) + max.squeeze(-1) - a[rows, target] 
        node.save(e/ sum_e, rows, target)
        return out
    @staticmethod
    def backward(node: Node, g):
        grad, rows, target = node.saved
        grad[rows, target] -= 1.0
        return (grad * g[:,xp.newaxis],)


            



        

    
