import os

# Use CuPy if available, otherwise Numpy
if os.environ.get("FEMTO_DEVICE", "gpu") == "gpu":
    try:
        import cupy as xp
        from cupy.lib.stride_tricks import sliding_window_view # for im2col operation
        xp.cuda.runtime.getDeviceCount()  # if no usable CUDA device
        
        GPU = True
    except Exception: # if xp.cuda.runtime.getDeviceCount() raises then numpy
        import numpy as xp
        from numpy.lib.stride_tricks import sliding_window_view
        GPU = False
    

else: # if FEMTO_DEVICE is set to something else than "gpu"
    import numpy as xp
    from numpy.lib.stride_tricks import sliding_window_view
    GPU = False

print(f"[femtotorch] backend: {'cupy (GPU)' if GPU else 'numpy (CPU)'}")


def to_cpu(a):
    """
    Examples:
    # eval in CIFAR_VGG.py: Ytest is numpy, pred.data may be cupy
    accuracy = (to_cpu(pred.data) == Ytest[:10000]).mean()

    """
    # needs a guard because cupy.asnumpy exists but numpy has no equivalent
    return xp.asnumpy(a) if GPU else a

def synchronize():
    # cupy kernels launch asynchronously; needed for honest benchmarks
    if GPU:
        xp.cuda.Device().synchronize()




