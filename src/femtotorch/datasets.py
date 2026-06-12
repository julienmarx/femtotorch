# Mnist specific, take advantage of its smallness to keep it in cache
# transform binary files into arrays of float

import gzip # compression library
import numpy as np
from pathlib import Path

def read_idx(path):
    with gzip.open(path, "rb") as f:
        header = int.from_bytes(f.read(4), "big") #reads the 4 bytes header of the file 
        ndim = header & 0xFF # extract the last byte of the header which is the ndim
        # read the next ndim bytes of the header which gives the size of each dim
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        # f.read() reads the rest of the data 
        # np.frombuffer(, np.uint8) convert the binary data in a 1d array of 8 bits unsigned integer
        # .reshape(dims) reshape typically the 1d vector of size 10000 * 28 * 28 into an array of shape (10000, 28, 28)
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


def load_mnist(data_dir):
    d = Path(data_dir) # cross platform path
    # flatten the 2d 28*28 digits into a 784 1D vector
    Xtrain = read_idx(d/"train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    Xtest = read_idx(d/"t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    # label target that is a
    Ytrain = read_idx(d/"train-labels-idx1-ubyte.gz")
    Ytest = read_idx(d/"t10k-labels-idx1-ubyte.gz")
    return Xtrain, Ytrain, Xtest, Ytest