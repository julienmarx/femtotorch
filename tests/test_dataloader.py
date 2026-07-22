import numpy as np
import pytest

from femtotorch.tensor import Tensor
from femtotorch.dataloader import Dataloader




def make_data(n, features=4):
    """Build aligned data: X[i, 0] == i and Y[i] == i.

    Encoding the index into both X and Y lets us verify that a sample's
    features stay paired with their label, even after the loader shuffles.
    """
    X = np.zeros((n, features), dtype=np.float64)
    X[:, 0] = np.arange(n)
    Y = np.arange(n)
    return X, Y


# --- batch counting ---------------------------------------------------------

def test_batch_count_when_evenly_divisible():
    X, Y = make_data(10)
    loader = Dataloader(X, Y, batch_size=5, shuffle=False)
    assert len(list(loader)) == 2


def test_batch_count_when_not_divisible():
    X, Y = make_data(10)
    loader = Dataloader(X, Y, batch_size=3, shuffle=False)
    assert len(list(loader)) == 4  # 3 + 3 + 3 + 1


def test_last_batch_is_smaller_when_not_divisible():
    X, Y = make_data(10)
    loader = Dataloader(X, Y, batch_size=3, shuffle=False)
    last_x, last_y = list(loader)[-1]
    assert last_x.data.shape[0] == 1
    assert last_y.shape[0] == 1


def test_batch_size_larger_than_dataset_gives_one_batch():
    X, Y = make_data(5)
    loader = Dataloader(X, Y, batch_size=10, shuffle=False)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][0].data.shape[0] == 5


# --- shapes and types -------------------------------------------------------

def test_batch_shapes():
    X, Y = make_data(8, features=4)
    loader = Dataloader(X, Y, batch_size=4, shuffle=False)
    for xb, yb in loader:
        assert xb.data.shape == (4, 4)
        assert yb.shape == (4,)


def test_x_is_wrapped_in_tensor_y_is_raw_ndarray():
    X, Y = make_data(6)
    loader = Dataloader(X, Y, batch_size=2, shuffle=False)
    xb, yb = next(iter(loader))
    assert isinstance(xb, Tensor)
    assert isinstance(yb, np.ndarray)


# --- ordering ---------------------------------------------------------------

def test_no_shuffle_preserves_order():
    X, Y = make_data(9)
    loader = Dataloader(X, Y, batch_size=4, shuffle=False)
    seen = np.concatenate([yb for _, yb in loader])
    np.testing.assert_array_equal(seen, np.arange(9))


def test_every_sample_seen_exactly_once_per_epoch():
    X, Y = make_data(10)
    loader = Dataloader(X, Y, batch_size=3, shuffle=True)
    seen = np.concatenate([yb for _, yb in loader])
    np.testing.assert_array_equal(np.sort(seen), np.arange(10))


def test_shuffle_changes_order():
    X, Y = make_data(100)
    np.random.seed(0)
    loader = Dataloader(X, Y, batch_size=10, shuffle=True)
    order = np.concatenate([yb for _, yb in loader])
    np.testing.assert_array_equal(np.sort(order), np.arange(100))  # still a permutation
    assert not np.array_equal(order, np.arange(100))               # but reordered


def test_reiterating_reshuffles():
    # __iter__ shuffles fresh each pass, so two epochs should differ.
    X, Y = make_data(100)
    np.random.seed(1) # to avoid false negative
    loader = Dataloader(X, Y, batch_size=10, shuffle=True)
    first = np.concatenate([yb for _, yb in loader])
    second = np.concatenate([yb for _, yb in loader])
    assert not np.array_equal(first, second)


# --- correctness of pairing -------------------------------------------------

def test_x_and_y_stay_aligned_after_shuffle():
    # X[i, 0] == Y[i] by construction, so the pairing must survive shuffling.
    X, Y = make_data(10)
    loader = Dataloader(X, Y, batch_size=3, shuffle=True)
    for xb, yb in loader:
        np.testing.assert_array_equal(xb.data[:, 0], yb)