import os
# os.environ["AF_JIT_KERNEL_TRACE"] = "stderr"
# os.environ["AF_TRACE"]="all"
# os.environ["AF_SHOW_LOAD_PATH"]="1"
# os.environ["AF_PRINT_ERRORS"]="1"
# os.environ["AF_BUILD_LIB_CUSTOM_PATH"]=...


from . import random
from . import fft
from . import distances 
from . import matrixprofile
from . import normalization

__all__ = ["random", "fft", "distances", "matrixprofile", "normalization"]


# direct imports
from ._device import *
from ._array_obj import *
from ._array_ops import *

__all__ += _device.__all__
__all__ += _array_obj.__all__
__all__ += _array_ops.__all__