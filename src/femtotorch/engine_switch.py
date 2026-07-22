import os

if os.environ.get("FEMTO_ENGINE", "v1") == "v2":
    from femtotorch.tensor import Tensor, no_grad
else:
    from femtotorch.TensorV2 import Tensor, no_grad

