import numpy as np
from unittest.mock import Mock

from femtotorch.engine_switch import Tensor
from femtotorch.optimizer import VanillaSGD


def make_param(data, grad):
    """Build a *real* leaf Tensor with a known gradient. This is not a test
    double: it returns a genuine Tensor, with .grad set by hand instead of
    populated by a backward pass (the arrange step the optimizer needs).
    """
    t = Tensor(data)
    t.grad = np.asarray(grad, dtype=np.float32)
    return t


# Tensor.data is float32, so compare with an explicit tolerance rather than
# relying on assert_allclose's float64 default rtol (1e-7).
TOL = dict(atol=1e-6)


# --- step() ---------------------------------------------------------------

def test_step_updates_single_param():
    p = make_param(data=[1.0], grad=[0.5])
    VanillaSGD([p], learning_rate=0.1).step()
    np.testing.assert_allclose(p.data, [0.95], **TOL)  # 1.0 - 0.1 * 0.5
    # **TOL = atol = 1e-6

def test_step_updates_all_params():
    a = make_param(data=[2.0], grad=[1.0])
    b = make_param(data=[-3.0], grad=[2.0])
    VanillaSGD([a, b], learning_rate=0.5).step()
    np.testing.assert_allclose(a.data, [1.5], **TOL)   # 2.0  - 0.5 * 1.0
    np.testing.assert_allclose(b.data, [-4.0], **TOL)  # -3.0 - 0.5 * 2.0


def test_step_is_elementwise_on_arrays():
    p = make_param(data=[[1.0, 2.0], [3.0, 4.0]],
                   grad=[[0.1, 0.2], [0.3, 0.4]])
    VanillaSGD([p], learning_rate=1.0).step()
    np.testing.assert_allclose(p.data, [[0.9, 1.8], [2.7, 3.6]], **TOL)


def test_step_scales_with_learning_rate():
    small = make_param(data=[0.0], grad=[1.0])
    big = make_param(data=[0.0], grad=[1.0])
    VanillaSGD([small], learning_rate=0.1).step()
    VanillaSGD([big], learning_rate=1.0).step()
    np.testing.assert_allclose(small.data, [-0.1], **TOL)
    np.testing.assert_allclose(big.data, [-1.0], **TOL)


def test_step_moves_opposite_to_gradient():
    pos = make_param(data=[0.0], grad=[1.0])   # positive grad -> param decreases
    neg = make_param(data=[0.0], grad=[-1.0])  # negative grad -> param increases
    VanillaSGD([pos, neg], learning_rate=0.1).step()
    assert pos.data[0] < 0.0
    assert neg.data[0] > 0.0


def test_step_with_zero_gradient_leaves_data_unchanged():
    p = make_param(data=[1.0, 2.0, 3.0], grad=[0.0, 0.0, 0.0])
    VanillaSGD([p], learning_rate=0.5).step()
    np.testing.assert_allclose(p.data, [1.0, 2.0, 3.0], **TOL)


def test_step_mutates_data_in_place():
    p = make_param(data=[1.0, 2.0], grad=[0.1, 0.2])
    before = p.data
    VanillaSGD([p], learning_rate=0.5).step()
    assert p.data is before  # `-=` mutates the array rather than rebinding it; "is" compares references


def test_repeated_steps_accumulate():
    p = make_param(data=[1.0], grad=[0.1])
    opt = VanillaSGD([p], learning_rate=0.1)
    opt.step()
    opt.step()
    np.testing.assert_allclose(p.data, [0.98], **TOL)  # two steps of -0.01


# --- zero_grad() ----------------------------------------------------------

# The optimizer's zero_grad does exactly one thing: forward the call to each
# parameter. That is an interaction, not a value computation, so a Mock is the
# right tool. Whether Tensor.zero_grad actually zeros .grad (and leaves .data
# untouched) is a property of Tensor and belongs in test_tensor.py.
def test_zero_grad_delegates_to_each_param():
    params = [Mock(), Mock()]
    VanillaSGD(params, learning_rate=0.1).zero_grad()
    for p in params:
        p.zero_grad.assert_called_once_with() #test if called one time


# --- construction ---------------------------------------------------------

def test_empty_parameter_list_is_safe():
    opt = VanillaSGD([], learning_rate=0.1)
    opt.step()       # should not raise
    opt.zero_grad()  # should not raise