import numpy as np
from femtotorch.loss import crossEntropy_MNIST, softmax
from  femtotorch.tensor import Tensor
# Initialize the random number generator
rng = np.random.default_rng()



class Layer():

    def __init__(self, nin, nout, activation = True):
        # std using He (if ReLU), std using Xavier-Glorot (if not)

        # without it training the MLP was impossible because the std exploded after just one hidden layer
        # which led to underflow in the loss function with exp(big_negative_number) = 0 by underflow
        # then the proba = 0 was fed into the gradient of crossentropy which has a log and log(0) = inf ;(
        std = np.sqrt(2.0/nin) if activation else np.sqrt(1/nin)
        self.W = Tensor(rng.standard_normal((nin, nout)) * std)
        # Original initialization: self.W = Tensor(rng.uniform(-1.0, 1.0, size= (nin, nout)))
        
        self.B = Tensor(np.zeros((1, nout)))
        self.activation = activation

    def __call__(self, X): # forward pass but with Layer(X)
        linear = (X @ self.W) + self.B # the bias vector self.B is broadcasted
        return linear.relu() if self.activation else linear
    
    def parameters(self):
        return [self.W, self.B] # python list returns W and B reference
    
    def zero_grad(self):
        self.W.zero_grad()
        self.B.zero_grad()

class MLP():

    def __init__(self, nin, nouts):

        sizes = [nin] + nouts # each contiguous couple of integers in this list gives info for layers
        self.layers = []

        # Iterate through to create all hidden layers with ReLU
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=True))
            
        # Append the final output layer without ReLU activation
        self.layers.append(Layer(sizes[-2], sizes[-1], activation=False))

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def parameters(self):
        # use list comprehension syntax to flatten all sublists [self.W, self.B] in an unique list without sublist
        return [p for layer in self.layers for p in layer.parameters()]
    

class Conv2d():

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # assuming relu activation
        fan_in = kernel_size * kernel_size * in_channels
        std = np.sqrt(2.0/(fan_in))
        self.W = Tensor(rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * std)
        
        self.B = Tensor(np.zeros((1, out_channels, 1, 1)))
        
        
    def __call__(self, X: Tensor): # X has shape (batch, (in) chanels, height, width)
        # kernel is square matrix
        out_height = ((X.data.shape[-2] - (self.kernel_size) + 2 * self.padding) // self.stride) + 1
        out_width = ((X.data.shape[-1] - (self.kernel_size) + 2 * self.padding) // self.stride) + 1

        batch = X.data.shape[-4]
        out_vals = [] 

        
        padded_in = X.pad_zeros((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))

        for b in range(batch): # considering X shape (batch, (in) chanels, height, width)
            for out_ch in range(self.out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        conv_sum = 0

                        for in_ch in range(self.in_channels):
                            input_window = padded_in[
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
    
    def zero_grad(self):
        self.W.zero_grad()
        self.B.zero_grad()


if __name__ == "__main__":
    convlayer = Conv2d()
    out = convlayer(Tensor(np.array([[[[1,2,3],
                                       [4,5,6],
                                       [7,8,9]]
                                       ]])))
    print(out)
    print(out.grad)

