
import os

# Use CuPy if available and not disabled, else fall back to Numpy
if os.environ.get("FEMTO_DEVICE", "gpu") == "gpu":
    try:
        import cupy as xp
        GPU = True
    except ImportError:
        import numpy as xp
        GPU = False

else:
    import numpy as xp
    GPU = False
    