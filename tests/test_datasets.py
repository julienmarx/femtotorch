import gzip
from pathlib import Path

import numpy as np
import pytest

from femtotorch.datasets import read_idx, one_hot, load_mnist


# --- helpers ---------------------------------------------------------------

def build_idx_bytes(array):
    """Serialize a uint8 array into the IDX binary format read_idx parses.

    Layout (big-endian): a 4-byte header whose last byte is the number of
    dimensions, then one 4-byte size per dimension, then the raw uint8 data.
    The 0x08 byte is the IDX type code for unsigned byte (what read_idx assumes).
    """
    array = np.asarray(array, dtype=np.uint8)
    header = bytes([0x00, 0x00, 0x08, array.ndim])
    dims = b"".join(int(d).to_bytes(4, "big") for d in array.shape)
    return header + dims + array.tobytes()


def write_gzipped_idx(path, array):
    """Write `array` as a gzipped IDX file at `path` (read_idx opens with gzip)."""
    path.write_bytes(gzip.compress(build_idx_bytes(array)))
    return path


# --- read_idx --------------------------------------------------------------

def test_read_idx_roundtrips_1d(tmp_path):
    original = np.array([0, 1, 2, 250, 255], dtype=np.uint8)
    path = write_gzipped_idx(tmp_path / "vec.gz", original)
    np.testing.assert_array_equal(read_idx(path), original)


def test_read_idx_recovers_shape_and_values_3d(tmp_path):
    # mimics the (n_images, rows, cols) layout of MNIST image files
    original = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = write_gzipped_idx(tmp_path / "imgs.gz", original)
    out = read_idx(path)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, original)


def test_read_idx_returns_uint8(tmp_path):
    path = write_gzipped_idx(tmp_path / "labels.gz", np.array([3, 1, 4], dtype=np.uint8))
    assert read_idx(path).dtype == np.uint8


# --- one_hot ---------------------------------------------------------------

def test_one_hot_shape():
    out = one_hot(np.array([0, 1, 2, 3]), num_classes=10)
    assert out.shape == (4, 10)


def test_one_hot_exactly_one_per_row_at_label_index():
    labels = np.array([0, 9, 4, 4])
    out = one_hot(labels, num_classes=10)
    # exactly one hot entry per row
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(4))
    # and it sits at the label's column
    np.testing.assert_array_equal(out.argmax(axis=1), labels)


def test_one_hot_is_float32():
    # downstream loss multiplies these masks with float32 softmax outputs
    assert one_hot(np.array([1, 2])).dtype == np.float32


def test_one_hot_respects_num_classes():
    out = one_hot(np.array([0, 2]), num_classes=3)
    np.testing.assert_array_equal(out, [[1, 0, 0], [0, 0, 1]])


def test_one_hot_defaults_to_ten_classes():
    assert one_hot(np.array([5])).shape == (1, 10)


# --- load_mnist (integration) ----------------------------------------------

# Touches the real dataset on disk; skip cleanly when it hasn't been downloaded.
mnist = pytest.mark.skipif(
    not Path("data/mnist/train-images-idx3-ubyte.gz").exists(),
    reason="data/mnist not present (run scripts/download_mnist.py)",
)


@mnist
def test_load_mnist_shapes_and_normalization():
    Xtrain, Ytrain, Xtest, Ytest = load_mnist("data/mnist")

    # images flattened to 784-vectors, standard MNIST split sizes
    assert Xtrain.shape == (60000, 784)
    assert Xtest.shape == (10000, 784)
    assert Ytrain.shape == (60000,)
    assert Ytest.shape == (10000,)

    # pixels normalized into [0, 1] as float32
    assert Xtrain.dtype == np.float32
    assert 0.0 <= Xtrain.min() and Xtrain.max() <= 1.0

    # labels are the 10 digit classes
    assert set(np.unique(Ytrain).tolist()) <= set(range(10))
