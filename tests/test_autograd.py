from femtotorch.tensor import Tensor
import numpy as np


def test_maxgradient():
    input = Tensor([[1,2],[3,4]])
    out = input.max()
    out.grad = 1
    out.backward()
    np.testing.assert_array_equal(input.grad, np.array([[0,0],[0,1]]))
    


if __name__ == "__main__":
    input = Tensor([[1,2],[3,4]])
    out = input.max()
    out.grad = 1
    out.backward()
    print(input.grad)

        