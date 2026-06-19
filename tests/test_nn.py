
import numpy as np
import pytest

from femtotorch.nn import Layer, MLP
from femtotorch.tensor import Tensor

# Seeded rng for reproducible *inputs* (the layers' own weights stay random)
rng = np.random.default_rng(0)


#------------------ #
# Layer


def test_layer_parameter_shapes():
    layer = Layer(4, 6)
    assert layer.W.data.shape == (4, 6)
    assert layer.B.data.shape == (1, 6)


def test_layer_bias_starts_at_zero():
    layer = Layer(4, 6)
    assert np.all(layer.B.data == 0.0)


def test_layer_parameters_returns_W_then_B():
    layer = Layer(3, 2)
    assert layer.parameters() == [layer.W, layer.B]


def test_he_init_std_for_relu_layer():
    # std should scale as sqrt(2 / nin) when activation is on (He)
    nin, nout = 1000, 200
    layer = Layer(nin, nout, activation=True)
    assert np.isclose(layer.W.data.std(), np.sqrt(2.0 / nin), rtol=0.1)


def test_xavier_init_std_for_linear_layer():
    # std should scale as sqrt(1 / nin) when activation is off
    nin, nout = 1000, 200
    layer = Layer(nin, nout, activation=False)
    assert np.isclose(layer.W.data.std(), np.sqrt(1.0 / nin), rtol=0.1)


def test_forward_output_shape():
    layer = Layer(4, 6)
    X = Tensor(rng.standard_normal((5, 4)))   # batch of 5
    assert layer(X).data.shape == (5, 6)


def test_relu_layer_output_is_non_negative():
    layer = Layer(4, 6, activation=True)
    X = Tensor(rng.standard_normal((5, 4)))
    assert np.all(layer(X).data >= 0.0)


def test_linear_layer_matches_manual_computation():
    layer = Layer(4, 6, activation=False)
    X = Tensor(rng.standard_normal((5, 4)))
    expected = X.data @ layer.W.data + layer.B.data   # bias broadcast over batch
    assert np.allclose(layer(X).data, expected)


def test_zero_grad_resets_gradients():
    layer = Layer(3, 2)
    layer.W.grad = np.ones_like(layer.W.data)
    layer.B.grad = np.ones_like(layer.B.data)
    layer.zero_grad()
    assert np.all(layer.W.grad == 0.0)
    assert np.all(layer.B.grad == 0.0)


#------------------ #
# MLP


def test_mlp_layer_count_equals_len_nouts():
    mlp = MLP(4, [8, 5, 3])
    assert len(mlp.layers) == 3


def test_mlp_layer_dimensions_chain():
    mlp = MLP(4, [8, 5, 3])
    shapes = [layer.W.data.shape for layer in mlp.layers]
    assert shapes == [(4, 8), (8, 5), (5, 3)]


def test_mlp_only_output_layer_is_linear():
    mlp = MLP(4, [8, 5, 3])
    assert [layer.activation for layer in mlp.layers] == [True, True, False]


def test_mlp_single_layer_is_linear():
    # nouts of length 1 => one layer, no hidden layers, no activation
    mlp = MLP(4, [3])
    assert len(mlp.layers) == 1
    assert mlp.layers[0].W.data.shape == (4, 3)
    assert mlp.layers[0].activation is False


def test_mlp_forward_output_shape():
    mlp = MLP(4, [8, 5, 3])
    X = Tensor(rng.standard_normal((10, 4)))
    assert mlp(X).data.shape == (10, 3)


def test_mlp_parameters_are_flattened():
    mlp = MLP(4, [8, 5, 3])
    params = mlp.parameters()
    assert len(params) == 2 * len(mlp.layers)            # W and B per layer
    assert all(isinstance(p, Tensor) for p in params)    # no nested sublists