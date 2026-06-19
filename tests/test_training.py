"""End-to-end training tests.

The unit tests pin down each primitive in isolation; these check that the
whole stack — MLP forward, softmax + cross-entropy, backward, and the SGD
update — actually composes into something that *learns*. The canonical sanity
check for any training loop is "can it overfit a tiny fixed dataset?". If the
gradients or the update step were wrong, loss would not drop to ~0.
"""

import numpy as np

from femtotorch.tensor import Tensor
from femtotorch.nn import MLP
from femtotorch.optimizer import VanillaSGD
from femtotorch.loss import softmax, crossEntropy_MNIST
from femtotorch.datasets import one_hot


def make_tiny_classification(n=8, dim=5, n_classes=3, seed=0):
    """A handful of random points with arbitrary labels — memorizable, not
    learnable in any general sense. Exactly what we want to overfit."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    Y = rng.integers(0, n_classes, size=n)
    return X, Y, n_classes


def train(model, X, Y, n_classes, lr=0.2, epochs=400):
    """Full-batch gradient descent. Returns the per-epoch loss history."""
    opt = VanillaSGD(model.parameters(), lr)
    targets = one_hot(Y, num_classes=n_classes)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        probs = softmax(model(Tensor(X)))
        loss = crossEntropy_MNIST(probs, Tensor(targets)).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.data))
    return history


def accuracy(model, X, Y):
    pred = softmax(model(Tensor(X))).argmax(axis=-1).data
    return (pred == Y).mean()


# --- overfitting -----------------------------------------------------------

def test_mlp_overfits_tiny_dataset():
    # Enough capacity + epochs that this succeeds regardless of the (unseeded)
    # weight init: a 32-unit hidden layer trivially memorizes 8 points.
    X, Y, n_classes = make_tiny_classification()
    model = MLP(X.shape[1], [32, n_classes])

    history = train(model, X, Y, n_classes)

    assert accuracy(model, X, Y) == 1.0          # perfectly memorized
    assert history[-1] < 0.05                    # loss driven near zero
    assert history[-1] < history[0]              # and it actually decreased


def test_loss_decreases_monotonically_on_average():
    # Not strictly monotonic step-to-step, but the start should dominate the end.
    X, Y, n_classes = make_tiny_classification(seed=1)
    model = MLP(X.shape[1], [32, n_classes])

    history = train(model, X, Y, n_classes)

    first_quarter = np.mean(history[: len(history) // 4])
    last_quarter = np.mean(history[-len(history) // 4 :])
    assert last_quarter < first_quarter


def test_single_linear_layer_learns_separable_problem():
    # No hidden layer: softmax-linear classifier on a linearly separable split
    # (label = sign of the first coordinate). Confirms the loop works without ReLU.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    Y = (X[:, 0] > 0).astype(int)  # 2 classes, linearly separable
    model = MLP(4, [2])  # single output layer (no activation)

    train(model, X, Y, n_classes=2, lr=0.5, epochs=300)

    assert accuracy(model, X, Y) >= 0.95
