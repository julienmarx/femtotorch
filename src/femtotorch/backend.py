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

print(f"[femtotorch] backend: {'cupy (GPU)' if GPU else 'numpy (CPU)'}")


def to_cpu(a):
    """
    Examples:
    # eval in CIFAR_VGG.py — Ytest is numpy, pred.data may be cupy
    accuracy = (to_cpu(pred.data) == Ytest[:10000]).mean()

    """
    # cupy.asnumpy exists; numpy has no equivalent, hence the guard
    return xp.asnumpy(a) if GPU else a

def synchronize():
    # cupy kernels launch asynchronously; needed for honest benchmarks
    if GPU:
        xp.cuda.Device().synchronize()


def memory_stats():
    """Returns a dict of memory numbers, or None on CPU."""
    if not GPU:
        return None
    pool = xp.get_default_memory_pool()
    free, total = xp.cuda.runtime.memGetInfo()   # bytes, whole device
    return {
        "pool_used":  pool.used_bytes(),    # bytes in live arrays right now
        "pool_total": pool.total_bytes(),   # bytes the pool holds (high-water mark)
        "dev_free":   free,                 # what the driver says is left on the card
        "dev_total":  total,
    }

