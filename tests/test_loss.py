import numpy as np

from femtotorch.tensor import Tensor
from femtotorch.loss import softmax, cross_entropy

RNG = np.random.default_rng(0)


# --- helpers ---------------------------------------------------------------

def np_softmax(x):
    """Plain-numpy reference softmax, for comparison."""
    shifted = (x - x.max(axis=-1, keepdims=True))
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def check_gradient(forward, x_np, eps=1e-5, rtol=1e-4, atol=1e-6):
    """Compare analytic backward grad against central finite differences.

    forward: takes a Tensor, returns a scalar Tensor (the loss).
    x_np:    numpy input we differentiate with respect to.
    """
    # analytic gradient
    x = Tensor(x_np.copy(), dtype = np.float64)
    forward(x).backward()
    analytic = np.array(x.grad, dtype = np.float64)

    # numerical gradient (central differences)
    numeric = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        plus, minus = x_np.astype(np.float64), x_np.astype(np.float64)
        plus[i] += eps
        minus[i] -= eps
        lp = float(forward(Tensor(plus)).data)
        lm = float(forward(Tensor(minus)).data)
        numeric[i] = (lp - lm) / (2 * eps)
        it.iternext()

    np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)


# --- softmax ---------------------------------------------------------------

def test_softmax_sums_to_one():
    x = Tensor(RNG.standard_normal((5, 10)))
    out = softmax(x).data
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(5), rtol=1e-6)


def test_softmax_all_positive():
    x = Tensor(RNG.standard_normal((5, 10)))
    assert (softmax(x).data > 0).all()


def test_softmax_matches_reference():
    x = RNG.standard_normal((3, 10))
    np.testing.assert_allclose(softmax(Tensor(x)).data, np_softmax(x), rtol=1e-6)


def test_softmax_shift_invariance():
    # adding a per-row constant must not change the output, and must not overflow
    x = RNG.standard_normal((4, 6))
    base = softmax(Tensor(x)).data
    shifted = softmax(Tensor(x + 100.0)).data
    np.testing.assert_allclose(base, shifted, rtol=1e-5)


def test_softmax_gradient():
    x = RNG.standard_normal((4, 10))

    def forward(t):
        s = softmax(t)
        return (s ** 2).sum()  # non-trivial scalar so the grad isn't identically 0

    check_gradient(forward, x)



# --- cross_entropy ----------------------------------------------------

def test_crossentropy_value():
    probs = np.array([[0.1, 0.7, 0.2],
                      [0.6, 0.3, 0.1]])
    target = np.array([[0, 1, 0],
                       [1, 0, 0]])
    loss = cross_entropy(Tensor(probs), Tensor(target)).data
    expected = -np.log([0.7, 0.6])
    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_crossentropy_confident_prediction_is_low():
    probs = np.array([[0.998, 0.001, 0.001]])
    target = np.array([[1, 0, 0]])
    loss = float(cross_entropy(Tensor(probs), Tensor(target)).data[0])
    assert loss < 0.01


def test_crossentropy_gradient():
    # positive inputs so log is defined; target is a fixed one-hot mask
    x = RNG.uniform(0.1, 0.9, size=(4, 10))
    onehot = np.zeros((4, 10))
    onehot[np.arange(4), RNG.integers(0, 10, size=4)] = 1.0
    target = Tensor(onehot)

    def forward(t):
        return cross_entropy(t, target).sum()

    check_gradient(forward, x)





