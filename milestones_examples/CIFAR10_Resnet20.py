
import femtotorch as ft
import time

"""
ResNet-20 for CIFAR-10 (He et al. 2015, "Deep Residual Learning for Image Recognition").
Depth = 6n + 2 with n = 3 blocks per stage:
    conv(16) -> [3 blocks @16, 32x32] -> [3 blocks @32, 16x16] -> [3 blocks @64, 8x8]
    -> global average pool -> linear(10)
"""