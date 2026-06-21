
import pytest
import numpy as np

from femtotorch.tensor import Tensor, unbroadcast

#-------------------------------#
# Construction & getters

def test_construction():
    t = Tensor([1, 2, 3])
    assert t.data.dtype == np.float32
    np.testing.assert_array_equal(t.data, [1, 2, 3])
    np.testing.assert_array_equal(t.grad, [0, 0, 0])     # grad starts at zero


def test_getters():
    t = Tensor([[1., 2., 3.], [4., 5., 6.]])
    assert t.shape() == (2, 3)
    assert t.size() == 6
    assert t.ndim() == 2
    assert Tensor(7.0).shape() == ()                     # scalar


def test_repr():
    assert "Tensor" in repr(Tensor([1., 2.]))


#-------------------------------#   
# Forward check

@pytest.fixture
def rng():
    """A seeded RNG so tests are deterministic (never flaky)."""
    return np.random.default_rng(0)

# Forward values tests

def test_add_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([5.0, 6.0, -2.0])
    # compared to == np.testing.assert_allclose takes into account floating piont imprecision
    np.testing.assert_allclose((a + b).data, [6.0, 8.0, 1.0])

def test_mul_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([5.0, 6.0, -2.0])
    np.testing.assert_allclose((a * b).data, [5.0, 12.0, -6.0])

def test_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    # use the underlying numpy array and numpy ops as a source of truth
    np.testing.assert_allclose((a @ b).data, a.data @ b.data) 

def test_sum_forward():
    a = Tensor([[1.0, 2.0, 3.0], [5.0, 6.0, -2.0]])
    np.testing.assert_allclose((a.sum()).data, 15.0)
    np.testing.assert_allclose((a.sum(keepdims=True)).data, [[15.0]])
    np.testing.assert_allclose((a.sum(axis = 0)).data, [6.0, 8.0, 1.0])
    np.testing.assert_allclose((a.sum(axis = 1)).data, [6.0, 9.0])

# shape of gradient matrices tests

@pytest.mark.parametrize("sa, sb", [
    ((2, 3), (2, 3)),    # same shape
    ((2, 3), (3,)),      # broadcast a row vector
    ((2, 3), (1, 3)),    # broadcast over rows
    ((2, 1), (1, 3)),    # broadcast both operands
])

def test_add_grad_shapes(rng, sa, sb):
    a = Tensor(rng.standard_normal(sa))
    b = Tensor(rng.standard_normal(sb))
    a.zero_grad()
    b.zero_grad()
    (a + b).backward()
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape







#-------------------------------#
# Gradient checking

def numerical_grad(f, x, eps=1e-3):
    """numerical approximation of the gradient: f(x+eps) - f(x - eps) / 2eps; O(esp^2) approximation error"""
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape): # np.ndindex to access every possible index of the array
        hi, lo = x.copy(), x.copy()
        hi[i] += eps
        lo[i] -= eps
        grad[i] = (f(hi) - f(lo)) / (2 * eps)
    return grad


def grad_check(f, *inputs, eps=1e-3, tol=1e-2):
    """
    *inputs: takes n argument and pack them in a tupple

    f: map that should always return just a scalar to avoid jacobiam matrix and have simple gradient,
    you can typically use .sum() to flatten the inputs into a scalar
    
    eps: infinitesimal value to compute the central difference derivative

    tol: tolerance of difference between gradient from autograd and from central difference derivative
    Assert .backward() agrees with finite differences for every input.


    """
    inputs = [np.asarray(x, dtype=np.float64) for x in inputs]

    # analytic: one backward pass fills the .grad of every leaf
    leaves = [Tensor(x, dtype=np.float64) for x in inputs]
    f(*leaves).backward()

    # forward value of f, as a plain float, from a list of input arrays
    def value(arrays):
        return float(f(*[Tensor(a, dtype=np.float64) for a in arrays]).data.sum())

    # numeric: differentiate w.r.t. each input in turn, holding the rest fixed
    for i, leaf in enumerate(leaves):
        def vary_input_i(xi, i=i):  # i=i binds the current loop index, to handle "Late Binding in Closures." 
            #(to avoid calling with a later value of i by the time the functions are called)
            args = list(inputs)
            args[i] = xi
            return value(args)
        expected = numerical_grad(vary_input_i, inputs[i], eps) # take advantange of vary_input_i closure
        np.testing.assert_allclose(leaf.grad, expected, rtol=tol, atol=tol,
                                   err_msg=f"gradient mismatch on input {i}")






#-------------------------------#
# Elementwise arithmetic





def test_add():
    a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
    np.testing.assert_allclose((Tensor(a) + Tensor(b)).data, [5, 7, 9])
    grad_check(lambda x, y: (x + y).sum(), a, b)


def test_mul():
    a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
    np.testing.assert_allclose((Tensor(a) * Tensor(b)).data, [4, 10, 18])
    grad_check(lambda x, y: (x * y).sum(), a, b)


def test_neg():
    a = np.array([1., -2., 3.])
    np.testing.assert_allclose((-Tensor(a)).data, [-1, 2, -3])
    grad_check(lambda x: (-x).sum(), a)


def test_sub():
    a, b = np.array([5., 7.]), np.array([1., 2.])
    np.testing.assert_allclose((Tensor(a) - Tensor(b)).data, [4, 5])
    grad_check(lambda x, y: (x - y).sum(), a, b)


def test_truediv():
    a, b = np.array([1., 4.]), np.array([2., 8.])        # positive denominator
    np.testing.assert_allclose((Tensor(a) / Tensor(b)).data, [0.5, 0.5], rtol=1e-5)
    grad_check(lambda x, y: (x / y).sum(), a, b)


def test_pow():
    a = np.array([1., 2., 3.])
    np.testing.assert_allclose((Tensor(a) ** 3).data, [1, 8, 27])
    grad_check(lambda x: (x ** 3).sum(), a)


def test_pow_rejects_tensor_exponent():
    with pytest.raises(AssertionError):
        Tensor([1., 2.]) ** Tensor([2.])

# test reflected operations
@pytest.mark.parametrize("op, expected", [
    pytest.param(lambda a: 2 + a, [3, 4], id="radd"),
    pytest.param(lambda a: 5 - a, [4, 3], id="rsub"),
    pytest.param(lambda a: 2 * a, [2, 4], id="rmul"),
    pytest.param(lambda a: 2 / a, [2, 1], id="rtruediv"),
])
def test_reflected_ops(op, expected):
    np.testing.assert_allclose(op(Tensor([1., 2.])).data, expected, rtol=1e-5)


#-------------------------------#
# Unary functions


def test_relu():
    a = np.array([-2., -0.5, 0.3, 1.5])                  # away from the kink at 0
    np.testing.assert_allclose(Tensor(a).relu().data, [0, 0, 0.3, 1.5], rtol=1e-5)
    grad_check(lambda x: x.relu().sum(), a)


def test_exp():
    a = np.array([-1., 0., 0.5, 1.])
    np.testing.assert_allclose(Tensor(a).exp().data, np.exp(a), rtol=1e-5)
    grad_check(lambda x: x.exp().sum(), a)


def test_log():
    a = np.array([0.5, 1., 2., np.e])                    # strictly positive
    np.testing.assert_allclose(Tensor(a).log().data, np.log(a), rtol=1e-5)
    grad_check(lambda x: x.log().sum(), a)


#-------------------------------#
# Reductions


def test_sum():
    a = np.array([[1., 2., 3.], [4., 5., 6.]])
    t = Tensor(a)
    np.testing.assert_allclose(t.sum().data, 21)
    np.testing.assert_allclose(t.sum(axis=0).data, [5, 7, 9])
    np.testing.assert_allclose(t.sum(axis=1, keepdims=True).data, [[6], [15]])
    grad_check(lambda x: x.sum(), a)
    grad_check(lambda x: (x.sum(axis=0) ** 2).sum(), a)                 # no keepdims
    grad_check(lambda x: (x.sum(axis=1, keepdims=True) ** 2).sum(), a)  # keepdims


def test_max():
    a = np.array([[1., 5., 2.], [8., 3., 4.]])           # distinct -> clean argmax
    t = Tensor(a)
    np.testing.assert_allclose(t.max().data, 8)
    np.testing.assert_allclose(t.max(axis=1).data, [5, 8])
    grad_check(lambda x: x.max().sum(), a)
    grad_check(lambda x: (x.max(axis=1) ** 2).sum(), a)


def test_max_ties(): # if there are multiple elements equals to max
    # tied maxima share the gradient equally (checked analytically; finite
    # differences can't see this cleanly)
    a = Tensor([3., 3., 1.])
    a.max().backward()
    np.testing.assert_allclose(a.grad, [0.5, 0.5, 0])

    b = Tensor([2., 2., 2.])
    b.max().backward()
    np.testing.assert_allclose(b.grad, [1/3, 1/3, 1/3])


def test_mean():
    a = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(Tensor(a).mean().data, 2.5)
    grad_check(lambda x: x.mean(), a)


#-------------------------------#
# Matmul

def test_matmul():
    a = np.array([[1., 2.], [3., 4.]])
    b = np.array([[5., 6.], [7., 8.]])
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, [[19, 22], [43, 50]])
    grad_check(lambda x, y: (x @ y).sum(), a, b)


def test_matmul_batched():
    # matching batch dims on both operands -> backward works as-is
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal((2, 4, 5))
    grad_check(lambda x, y: (x @ y).sum(), a, b)


def test_matmul_requires_2d():
    with pytest.raises(AssertionError):
        Tensor([1., 2., 3.]) @ Tensor([1., 2., 3.])


#-------------------------------#
# Indexing

def test_getitem():
    a = np.array([10., 20., 30.])
    t = Tensor(a)
    np.testing.assert_allclose(t[1].data, 20)
    np.testing.assert_allclose(t[[0, 2]].data, [10, 30])
    grad_check(lambda x: (x[1:] ** 2).sum(), a)


def test_getitem_repeated_indices():
    # the np.add.at path: a repeated index must accumulate gradient
    a = Tensor([1., 2., 3.])
    a[[0, 0, 2]].sum().backward()
    np.testing.assert_allclose(a.grad, [2, 0, 1])


#-------------------------------#
# Pad

def test_pad_zeros_forward():
    a = Tensor([[1., 2.], [3., 4.]])
    out = a.pad_zeros((1, 1), (1, 1))                          # 1 of zeros all around
    assert out.shape() == (4, 4)
    np.testing.assert_allclose(out.data, np.pad(a.data, ((1, 1), (1, 1))))


def test_pad_zeros_asymmetric_forward():
    a = Tensor([[1., 2., 3.]])
    out = a.pad_zeros((2, 0), (1, 3))                          # different front/back per axis
    assert out.shape() == (3, 7)
    np.testing.assert_allclose(out.data, np.pad(a.data, ((2, 0), (1, 3))))


def test_pad_zeros_prev_is_the_real_parent():
    a = Tensor([[1., 2.], [3., 4.]])
    assert a.pad_zeros((1, 1), (1, 1))._prev == {a}


def test_pad_zeros_grad():
    a = np.array([[1., 2.], [3., 4.]])
    grad_check(lambda x: (x.pad_zeros((1, 1), (1, 1)) ** 2).sum(), a)
    grad_check(lambda x: (x.pad_zeros((2, 0), (0, 3)) ** 2).sum(), a)   # asymmetric


def test_pad_zeros_grad_drops_border_keeps_interior():
    # f = sum(pad^2): interior grad is 2*x, the zero border contributes nothing
    a = Tensor([[1., 2.], [3., 4.]])
    (a.pad_zeros((1, 1), (1, 1)) ** 2).sum().backward()
    assert a.grad.shape == (2, 2)                              # routed back to original shape
    np.testing.assert_allclose(a.grad, 2 * a.data)


def test_pad_zeros_of_intermediate_propagates_further():
    # pad sits mid-graph; gradient must reach the leaf below it
    a = Tensor([[1., 2., 3.]])
    (a * 2).pad_zeros((1, 0), (0, 2)).sum().backward()
    np.testing.assert_allclose(a.grad, [[2, 2, 2]])            # d/da of sum(2a) = 2


def test_pad_zeros_no_padding_is_identity():
    a = Tensor([[1., 2.], [3., 4.]])
    out = a.pad_zeros((0, 0), (0, 0))
    np.testing.assert_allclose(out.data, a.data)
    (out ** 2).sum().backward()
    np.testing.assert_allclose(a.grad, 2 * a.data)


#-------------------------------#
# Reshape

def test_reshape_forward():
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    out = a.reshape(3, 2)
    assert out.shape() == (3, 2)
    np.testing.assert_allclose(out.data, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(out.reshape(6).data, [1, 2, 3, 4, 5, 6])   # flatten


def test_reshape_prev_is_the_real_parent():
    # _prev must hold the source tensor itself, not slices from iterating it
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    out = a.reshape(6)
    assert out._prev == {a}


def test_reshape_grad():
    a = np.array([[1., 2., 3.], [4., 5., 6.]])
    grad_check(lambda x: (x.reshape(3, 2) ** 2).sum(), a)
    grad_check(lambda x: (x.reshape(6) ** 2).sum(), a)


def test_reshape_grad_routes_back_to_original_shape():
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    (a.reshape(6) ** 2).sum().backward()              # grad = 2*data, in a's shape
    assert a.grad.shape == (2, 3)
    np.testing.assert_allclose(a.grad, 2 * a.data)


def test_reshape_of_intermediate_propagates_further():
    # reshape sits mid-graph; gradient must keep flowing to the leaves below it
    x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    w = Tensor([[1., 1., 1.], [1., 1., 1.]])
    ((x * w).reshape(6)).sum().backward()
    np.testing.assert_allclose(x.grad, [[1, 1, 1], [1, 1, 1]])   # d/dx = w
    np.testing.assert_allclose(w.grad, x.data)                   # d/dw = x


def test_reshape_of_scalar_intermediate():
    # 0-d intermediate can't be iterated; the parent must still be wired up
    x = Tensor([1., 2., 3.])
    x.sum().reshape(1).backward()
    np.testing.assert_allclose(x.grad, [1, 1, 1])


#-------------------------------#
# Stack

def test_stack_forward():
    a = Tensor([1., 2., 3.])
    b = Tensor([4., 5., 6.])
    out = Tensor.stack([a, b])
    assert out.shape() == (2, 3)                          # new leading axis
    np.testing.assert_allclose(out.data, [[1, 2, 3], [4, 5, 6]])


def test_stack_grad():
    # each input's grad is its own slice of out.grad, not a broadcast scalar
    a = np.array([1., 2., 3.])
    b = np.array([4., 5., 6.])
    c = np.array([-1., 0., 7.])
    grad_check(lambda *xs: Tensor.stack(list(xs)).sum(), a, b, c)
    # weight the slices differently so a bug that ignores position is caught
    grad_check(lambda *xs: (Tensor.stack(list(xs)) ** 2).sum(), a, b, c)


def test_stack_grad_values():
    a = Tensor([1., 2., 3.])
    b = Tensor([4., 5., 6.])
    (Tensor.stack([a, b]) ** 2).sum().backward()          # f = sum(stack^2) -> grad = 2*data
    np.testing.assert_allclose(a.grad, [2, 4, 6])
    np.testing.assert_allclose(b.grad, [8, 10, 12])


def test_stack_scalars():
    a, b = Tensor(2.), Tensor(5.)
    out = Tensor.stack([a, b])
    assert out.shape() == (2,)
    out.sum().backward()
    np.testing.assert_allclose(a.grad, 1)
    np.testing.assert_allclose(b.grad, 1)


def test_stack_repeated_input_accumulates():
    # the same tensor stacked twice must accumulate gradient from both slices
    a = Tensor([1., 2.])
    Tensor.stack([a, a]).sum().backward()
    np.testing.assert_allclose(a.grad, [2, 2])


#-------------------------------#
# Inference helper

def test_argmax():
    a = Tensor([[1., 9., 3.], [7., 2., 5.]])
    np.testing.assert_array_equal(a.argmax(axis=1).data, [1, 0])


#-------------------------------#
# Broadcasting (unbroadcast)

def test_unbroadcast():
    g = np.ones((4, 3))
    # extra leading dim collapses; rank drops to match (3,)
    np.testing.assert_allclose(unbroadcast(g, (3,)), [4, 4, 4])
    assert unbroadcast(g, (3,)).shape == (3,)
    # size-1 dim collapses but rank is kept to match (1, 3)
    np.testing.assert_allclose(unbroadcast(g, (1, 3)), [[4, 4, 4]])
    assert unbroadcast(g, (1, 3)).shape == (1, 3)


def test_add_broadcasting():
    # the two shapes a bias takes in practice: (D,) and (1, D)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 3))
    grad_check(lambda w, b: (w + b).sum(), W, rng.standard_normal((3,)))
    grad_check(lambda w, b: (w + b).sum(), W, rng.standard_normal((1, 3)))


def test_mul_broadcasting():
    rng = np.random.default_rng(0)
    grad_check(lambda w, s: (w * s).sum(), rng.standard_normal((4, 3)), rng.standard_normal((3,)))


#-------------------------------#
# Autograd engine

def test_zero_grad():
    x = Tensor([1., 2.])
    (x ** 2).sum().backward()
    assert np.any(x.grad != 0)
    x.zero_grad()
    np.testing.assert_array_equal(x.grad, [0, 0])


def test_backward_seeds_root():
    x = Tensor([1., 2., 3.])
    x.backward()                                          # root grad seeded to ones
    np.testing.assert_array_equal(x.grad, [1, 1, 1])


def test_backward_accumulates_shared_node():
    x = Tensor([3.])
    (x + x).backward()                                    # y = 2x -> dy/dx = 2
    np.testing.assert_allclose(x.grad, [2])


def test_backward_diamond():
    x = Tensor([4.])
    f = x * x + x                                         # f = x^2 + x -> f' = 2x + 1
    f.backward()
    np.testing.assert_allclose(f.data, [20])
    np.testing.assert_allclose(x.grad, [9])


#-------------------------------#
# Integration

def test_linear_layer():
    # x @ W + b, squared: matmul, bias broadcast, pow and sum together
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    W = rng.standard_normal((3, 2))
    b = rng.standard_normal((2,))
    grad_check(lambda X, Ww, bb: ((X @ Ww + bb) ** 2).sum(), x, W, b)



def test_mean_over_axis():
    a = Tensor([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(a.mean(axis=0).data, [2, 3])
    np.testing.assert_allclose(a.mean(axis=1).data, [1.5, 3.5])  # worth adding the other axis too


# --------------------------------------------------------------------------- #
# KNOWN LIMITATION: matmul backward has no unbroadcast over batch dims.
#   (B, T, d) @ (d, h) broadcasts the 2D weight in the forward pass, but
#   other.grad += swapaxes(self) @ out.grad is (B, d, h) and can't accumulate
#   into the (d, h) weight -> ValueError. This bites in the Transformer chapter
#   (nn.Linear-style x @ W with 3D x). Fix: unbroadcast both matmul grads.
# --------------------------------------------------------------------------- #


"""

@pytest.mark.xfail(reason="matmul backward lacks unbroadcast over batch dims",
                   strict=False, raises=ValueError)
def test_matmul_broadcast_operand():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4))   # (B, T, d)
    W = rng.standard_normal((4, 5))      # (d, h) shared across the batch
    grad_check(lambda X, Ww: (X @ Ww).sum(), x, W)

"""