# keeping module
from . import random
from . import fft
from . import distances 
<<<<<<< HEAD
from . import matrixprofile

__all__ = ["random", "fft", "distances", "matrixprofile"]
=======

__all__ = ["random", "fft", "distances"]
>>>>>>> master


# direct imports
from ._device import *
from ._array_obj import *
from ._array_ops import *

__all__ += _device.__all__
__all__ += _array_obj.__all__
__all__ += _array_ops.__all__