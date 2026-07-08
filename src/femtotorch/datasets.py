# Mnist specific, take advantage of its smallness to keep it in cache
# transform binary files into arrays of float

import gzip # compression library
import pickle # cifar-10 python batches are pickled dicts
import tarfile # to unpack cifar-10-python.tar.gz
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

def one_hot(labels, num_classes = 10):
    labels = np.asarray(labels)
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32) # 60000 * 10 matrix
    out[np.arange(labels.shape[0]), labels] = 1 # [[0, 1, ..., 59999], [1, 9, .., 3]] assigns a 1 in each row
    return out

def load_mnist(data_dir):
    d = Path(data_dir) # cross platform path
    # flatten the 2d 28*28 digits into a 784 1D vector
    Xtrain = read_idx(d/"train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    Xtest = read_idx(d/"t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    # label target that is a
    Ytrain = read_idx(d/"train-labels-idx1-ubyte.gz")
    Ytest = read_idx(d/"t10k-labels-idx1-ubyte.gz")
    return Xtrain, Ytrain, Xtest, Ytest

def _unpickle_cifar_batch(path):
    # each cifar-10 python batch is a dict pickled by python 2, so we must tell
    # pickle to decode the byte-string keys with encoding="bytes", otherwise it errors
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    # b'data'   -> uint8 array of shape (10000, 3072) : 1024 R + 1024 G + 1024 B per row (planar, channel-first)
    # b'labels' -> python list of 10000 ints in [0, 9]
    return batch[b"data"], np.asarray(batch[b"labels"])

def load_cifar10(data_dir):
    d = Path(data_dir)
    batches_dir = d / "cifar-10-batches-py" # the folder created when the tarball is extracted

    # auto-extract the tarball the first time, so the caller only has to run the download script.
    # (load_mnist reads .gz directly; cifar ships as a single .tar.gz, so we unpack it once)
    if not batches_dir.exists():
        with tarfile.open(d / "cifar-10-python.tar.gz", "r:gz") as tar:
            tar.extractall(d)

    # the 50000 training images are split across 5 files; stack them into one array
    train_data, train_labels = [], []
    for i in range(1, 6):
        data, labels = _unpickle_cifar_batch(batches_dir / f"data_batch_{i}")
        train_data.append(data)
        train_labels.append(labels)
    Xtrain = np.concatenate(train_data).astype(np.float32) / 255.0 # (50000, 3072) scaled to [0, 1]
    Ytrain = np.concatenate(train_labels)

    # the 10000 test images live in a single file
    test_data, test_labels = _unpickle_cifar_batch(batches_dir / "test_batch")
    Xtest = test_data.astype(np.float32) / 255.0 # (10000, 3072)
    Ytest = test_labels

    # kept flat like load_mnist: the model reshapes each row to (3, 32, 32) in its forward pass
    return Xtrain, Ytrain, Xtest, Ytest
