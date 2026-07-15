import numpy as np
from  femtotorch.tensor import Tensor
from abc import ABC, abstractmethod


# Initialize the random number generator
rng = np.random.default_rng()


class Module(ABC):

    @abstractmethod
    def forward(self, X):
        """Compute output"""

    def parameters(self):
        """returns list of weights (Tensor objects)"""
        return []

    def __call__(self, X: Tensor):
        return self.forward(X)
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None # lazy initialization

class Layer(Module):

    def __init__(self, nin, nout, activation = True):
        # std using He (if ReLU), std using Xavier-Glorot (if not)

        # without it training the MLP was impossible because the std exploded after just one hidden layer
        # which led to underflow in the loss function with exp(big_negative_number) = 0 by underflow
        # then the proba = 0 was fed into the gradient of crossentropy which has a log and log(0) = inf ;(
        std = np.sqrt(2.0/nin) if activation else np.sqrt(1/nin)
        self.W = Tensor(rng.standard_normal((nin, nout)) * std, dtype=np.float32)
        # Problematic original initialization: rng.uniform(-1.0, 1.0, size= (nin, nout)))
        
        self.B = Tensor(np.zeros((1, nout)), dtype=np.float32)
        self.activation = activation

    def forward(self, X): # forward pass but with Layer(X)
        linear = (X @ self.W) + self.B # the bias vector self.B is broadcasted
        return linear.relu() if self.activation else linear
    
    def parameters(self):
        return [self.W, self.B] # python list returns W and B reference

class MLP(Module):

    def __init__(self, nin, nouts):

        sizes = [nin] + nouts # each contiguous couple of integers in this list gives info for layers
        self.layers = []

        # Iterate through to create all hidden layers with ReLU
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=True))
            
        # Append the final output layer without ReLU activation
        self.layers.append(Layer(sizes[-2], sizes[-1], activation=False))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def parameters(self):
        # use list comprehension syntax to flatten all sublists [self.W, self.B] in an unique list without sublist
        return [p for layer in self.layers for p in layer.parameters()]
    
def conv_out_size(shape, kernel_size, stride, padding):
    """
    in_size is (height, width)
    """

    out_height = ((shape[-2] - (kernel_size) + 2 * padding) // stride) + 1
    out_width = ((shape[-1] - (kernel_size) + 2 * padding) // stride) + 1

    return (out_height, out_width)



class VanillaConv2d(Module):
    """
    Version 1 of convolution layer inneficient but pedagogical using python loops and operating at scalar operations/tensor scale.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # assuming relu activation
        fan_in = kernel_size * kernel_size * in_channels
        std = np.sqrt(2.0/(fan_in))
        self.W = Tensor(rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * std, dtype=np.float32)
        self.B = Tensor(np.zeros((1, out_channels, 1, 1)), dtype=np.float32)
        
        
    def forward(self, X: Tensor): # X has shape (batch, (in) chanels, height, width)
        # kernel is square matrix

        out_size = conv_out_size(X.shape, self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]

        batch = X.shape[-4]
        out_vals = [] 

        
        padded_X = X.pad_zeros((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))

        for b in range(batch): # considering X shape (batch, (in) chanels, height, width)
            for out_ch in range(self.out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        conv_sum = 0

                        for in_ch in range(self.in_channels):
                            input_window = padded_X[
                                                    b,
                                                    in_ch,
                                                    in_h_start:in_h_start + self.kernel_size,
                                                    in_w_start: in_w_start + self.kernel_size,                            
                                                    ]

                            weight_val = self.W[out_ch, in_ch, :, :]
                            conv_sum += (input_window * weight_val).sum()
                        out_vals.append(conv_sum)

        flatten_feature_map = Tensor.stack(out_vals)
        feature_map = flatten_feature_map.reshape(batch, self.out_channels, out_height, out_width)
        out = feature_map + self.B

        return out
    
    def parameters(self):
        return [self.W, self.B] # python list returns W and B reference
    

    def size_map(self, in_height, in_width):
        out_size = conv_out_size((in_height, in_width), self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]
        return self.out_channels * out_height * out_width



class Conv2d(Module):
    """
    Version 2, constructing im2col abstraction with python, way faster.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # fan_in = number of weights
        self.fan_in = kernel_size * kernel_size * in_channels

        std = np.sqrt(2.0/(self.fan_in))
        self.W = Tensor(rng.standard_normal((self.fan_in, self.out_channels)) * std, dtype=np.float32) # the weights are flatten to addapt to
        self.B = Tensor(np.zeros((1, out_channels, 1, 1)), dtype=np.float32)

        

    def forward(self, X: Tensor): # X has shape (batch, (in) chanels, height, width)
        # kernel is square matrix
        out_size = conv_out_size(X.shape, self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]
        
        batch = X.shape[-4]
        patches_list = [] # will store batch of input batch that will be stack and flatten to use im2col
        padded_X = X.pad_zeros((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))

        for out_h in range(out_height):
            for out_w in range(out_width):
                patch_h_start = out_h * self.stride
                patch_w_start = out_w * self.stride

                patch_ij = padded_X[
                                    :,
                                    :,
                                    patch_h_start : patch_h_start + self.kernel_size, # i:j takes position [i, i + 1, ..., j - 1]
                                    patch_w_start : patch_w_start + self.kernel_size
                                    ] # shape of patch_ij = (batch, (in)channels, kernel_size, kernel_size)
                
                patches_list.append(patch_ij)
        # patches_list becomes a Tensor of shape (out_height * out_width, batch, (in)channesl, kernel_size, kernel_size) 
        im2col = Tensor.stack(patches_list)
        im2col = im2col.swapaxes(0, 1) # swap to have (batch, out_height * out_width, (in)channesl, kernel_size, kernel_size) 
        im2col = im2col.reshape(batch * out_height * out_width, self.fan_in) 
        
        # the main gain compared to vanilla conv2d here all the operations are vectorized
        # just one big node matmul instead of numerous node created inside the nested loops for each entry-wise multiplication operation
        out_flatten = im2col @ self.W # shape is (batch * out_height * out_width, out_channels)
        feature_map = out_flatten.reshape(batch, out_height, out_width, self.out_channels)

        feature_map = feature_map.swapaxes(2, 3).swapaxes(1, 2) # shape (batch, out_channels, out_height, out_width)

        out = feature_map + self.B 

        return out # shape (batch, out_channels, out_height, out_width)
    
    def parameters(self):
        return [self.W, self.B] # python list returns W and B reference
    

    def size_map(self, in_height, in_width):
        out_size = conv_out_size((in_height, in_width), self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]
        return self.out_channels * out_height * out_width
    
    def out_height(self, in_height):
        out_height = conv_out_size((in_height, in_height), self.kernel_size, self.stride, self.padding)[-2]
        return out_height
    
    def out_width(self, in_width):
        out_width = conv_out_size((in_width, in_width), self.kernel_size, self.stride, self.padding)[-1]
        return out_width
    

class OptiConv2d(Module):
    """
    Optimized version, reducing as much as possible python overhead, using vectorized and fused operations (im2col reshape and transform notably)
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.fan_in = kernel_size * kernel_size * in_channels

        std = np.sqrt(2.0/(self.fan_in))
        self.W = Tensor(rng.standard_normal((self.fan_in, self.out_channels)) * std, dtype=np.float32) # the weights are flatten to addapt to
        self.B = Tensor(np.zeros((1, out_channels, 1, 1)), dtype=np.float32) if self.bias else None
        

    def forward(self, X: Tensor): # X has shape (batch, (in) chanels, height, width)
        
        out_size = conv_out_size(X.shape, self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]

        batch = X.shape[-4]
        padded_X = X.pad_zeros((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))

        # the loop with lots of python overhead and node in the gradient graph is replace by a vectorized im2col tensor operation

        im2col = padded_X.im2col(kernel_size=self.kernel_size, stride=self.stride)
        
        out_flatten = im2col @ self.W # (batch * out_width * out_height, fan_in) @ (fan_in, out_channels) =(batch * out_height * out_width, out_channels)
        feature_map = out_flatten.reshape(batch, out_height, out_width, self.out_channels)

        feature_map = feature_map.swapaxes(2, 3).swapaxes(1, 2) # shape (batch, out_channels, out_height, out_width)
        if self.bias:
            out = feature_map + self.B 
        else:
            out = feature_map
        
        return out # shape (batch, out_channels, out_height, out_width)
    
    def parameters(self):
        return [self.W, self.B] if self.bias else [self.W]
    

    def size_map(self, in_height, in_width):
        out_size = conv_out_size((in_height, in_width), self.kernel_size, self.stride, self.padding)
        out_height = out_size[0]
        out_width = out_size[1]
        return self.out_channels * out_height * out_width
    
    def out_height(self, in_height):
        out_height = conv_out_size((in_height, in_height), self.kernel_size, self.stride, self.padding)[-2]
        return out_height
    
    def out_width(self, in_width):
        out_width = conv_out_size((in_width, in_width), self.kernel_size, self.stride, self.padding)[-1]
        return out_width

class MaxPool2d(Module):
    """
    MaxPool layer with stride == kernel_size
    The windows tile the input with no overlap, so the whole layer is a reshape
    into windows followed by a max over the two in-window axes.
    """
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, X: Tensor): # X has shape (batch, in_channels, height, width)
        batch, in_channels, height, width = X.shape
        
        assert height % self.kernel_size == 0 and width % self.kernel_size == 0,\
        f"height or width is not a multiple of kernel_size"


        window = X.reshape(batch,
                            in_channels,
                            height // self.kernel_size,
                            self.kernel_size,
                            width // self.kernel_size,
                            self.kernel_size
                            )
        out = window.max(axis=(3, 5), keepdims=False)
        return out # shape (batch, in_channels, out_height, out_width)
    
    def out_height(self, in_height):
        out_height = in_height // self.kernel_size
        return out_height
    
    def out_width(self, in_width):
        out_width = in_width // self.kernel_size
        return out_width


class BatchNorm2d(Module):
    """
    After a Conv2d, normalize outputs in a standard normal distribution
    then learn to recenter and scale the distribution of the output values
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = Tensor(np.ones(shape=(1, num_features, 1, 1)), dtype = np.float32)
        self.beta = Tensor(np.zeros(shape=(1, num_features, 1, 1)), dtype = np.float32)
        self.running_mean = np.zeros(shape=(1, num_features, 1, 1), dtype= np.float32)
        self.running_var = np.ones(shape=(1, num_features, 1, 1), dtype=np.float32)
        self.momentum = momentum
        self.training = True

    def set_training(self, training = True):
        self.training = training

    def forward(self, X: Tensor): # X.shape is (batch, in_channel, height, width)
        

        if self.training:

            mu = X.mean(axis=(0, 2, 3), keepdims = True) # per in_channel mean
            var = ((X - mu)**2).mean(axis=(0, 2, 3), keepdims = True)

            #running var and mu are numpy arrays to avoid unnecessary graph construction
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            z_score = (X - mu) / ((var + self.eps) ** 0.5)
            
        else:
            z_score = (X - self.running_mean) / ((self.running_var + self.eps) ** 0.5)

        out = z_score * self.gamma + self.beta 

        return out
    
    def parameters(self):
        return [self.gamma, self.beta]
