import os

# Use CuPy if available and not disabled, else fall back to Numpy
if os.environ.get("FEMTO_DEVICE", "gpu") == "gpu":
    try:
        import cupy as xp
        from cupy.lib.stride_tricks import sliding_window_view # for im2col
        xp.cuda.runtime.getDeviceCount()  # raises if no usable CUDA device
        
        GPU = True
    except Exception:
        import numpy as xp
        from numpy.lib.stride_tricks import sliding_window_view
        GPU = False
    

else:
    import numpy as xp
    from numpy.lib.stride_tricks import sliding_window_view
    GPU = False


def to_cpu(a):
    """
    Examples:
    # eval in CIFAR_VGG.py — Ytest is numpy, pred.data may be cupy
    accuracy = (to_cpu(pred.data) == Ytest[:10000]).mean()

    # weights.py save — np.savez can't accept cupy arrays
    array_dict = {f"p{i}": to_cpu(arr.data) for i, arr in enumerate(parameters_list)}

    # tests — np.testing.assert_allclose can't accept cupy arrays either
    np.testing.assert_allclose(to_cpu(t.data), expected)

    """
    # cupy.asnumpy exists; numpy has no equivalent, hence the guard
    return xp.asnumpy(a) if GPU else a

def synchronize():
    # cupy kernels launch asynchronously; needed for honest benchmarks
    if GPU:
        xp.cuda.Device().synchronize()

