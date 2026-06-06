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
        self.grad = 0
        self.data = np.asarray(data, dtype=np.float32) # the prefix "as"array avoid copy when data is already well formated
        self._prev = set(_prev)
        self._backward = lambda: None # by default a dummy function returning None
        # for leaf nodes in the autograd engine
    if __name__ == "__main__":
        print("Let's go !")