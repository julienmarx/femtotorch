![CI](https://github.com/julienmarx/femtotorch/actions/workflows/ci.yml/badge.svg)


A coherent NumPy/CuPy deep-learning curriculum where every later model imports the previous one: Ch0 tensor autograd, Ch1 MLP/residual nets,  Ch2 CNN/CIFAR, Ch3 decoder Transformer/Tiny Shakespeare

## Setup

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Install uv if you don't have it yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the project dependencies:

```bash
uv sync              # cpu, numpy only
uv sync --extra gpu  # adds cupy, needs an nvidia gpu with a cuda 12 driver
```

The gpu is used automatically when available; `FEMTO_DEVICE=cpu` forces numpy.

Download the datasets once:

```bash
uv run scripts/download_cifar10.py   # to data/cifar10
uv run scripts/download_mnist.py     # to data/mnist
uv run scripts/download_fashion.py   # to data/fashion_mnist
```

then train:
  
```bash
uv run milestones_examples/MNIST_MLP.py
uv run milestones_examples/CIFAR_VGG.py
```

Chapter 0 : autograd engine

Chapter 1 : deep neural net
with example MNIST and Fashion Mnist

Chapter 2: Convolution network
with example vision on CIFAR 10


Available soon s:
Chapter 3: Attention
 with example miniGPT with Tiny Shakespeare
