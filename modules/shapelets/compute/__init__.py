from __future__ import annotations
import os


# os.environ["AF_JIT_KERNEL_TRACE"] = "stderr"
# os.environ["AF_TRACE"]="all"
# os.environ["AF_SHOW_LOAD_PATH"]="1"
# os.environ["AF_PRINT_ERRORS"]="1"

# current location
compute_dir = os.path.abspath(os.path.dirname(__file__)) 

# look if the production directory exists...
__library_dir__ = os.path.join(compute_dir, '.libs')

if not os.path.exists(__library_dir__):
    raise RuntimeError("No valid location can be stablished for native libraries. " + __library_dir__)

# Let AF know where the RT libraries are
os.environ["AF_PATH"] = __library_dir__
os.environ["AF_BUILD_LIB_CUSTOM_PATH"] = __library_dir__

if os.name == 'nt':
    import warnings 
    import glob
    import sys 
    try:
        from ctypes import PyDLL
        # Ensure present folder is present in the path
        os.environ['PATH'] = __library_dir__ + ';' + compute_dir + ';' + os.environ['PATH']
    except:
        warnings.warn('Unable to perform initialization', stacklevel=2)    
    else:
        filenames = []
        pyver = sys.version_info 
        has_winmode = pyver.major >= 3 and pyver.minor >= 8
        # For some reason, we are forced to load the library using 
        # the semantics of LoadLibrary and not LoadLibraryEx in 
        # windows system.  It requires more investigation but it may 
        # have something to do with the fact we are loading arrayfire 
        for filename in glob.glob(os.path.join(compute_dir, '_pygauss*.pyd')): 
            if has_winmode:
                PyDLL(filename, winmode= 0)  
            else:
                PyDLL(filename)
                
            filenames.append(filename)

        if len(filenames) > 1:
            warnings.warn("Loaded more than one pyd library in compute folder: %s" % repr(filenames), stacklevel=2)

del compute_dir

from . import _pygauss
__af_version__ = _pygauss.af_version()

from . import random
from . import fft
from . import distances 
from . import matrixprofile
from . import normalization
from . import dimensionality
from . import clustering

__all__ = ["random", "fft", "distances", "matrixprofile", "normalization", "dimensionality", "clustering"]

# direct imports
from . import _device 
from ._device import *

from . import _array_obj 
from ._array_obj import *

from . import _algorithms 
from ._algorithms import *

from . import _logic 
from ._logic import *

from . import _math 
from ._math import *

from . import _construction 
from ._construction import *

from . import _extract_transform 
from ._extract_transform import *

from . import _linear_algebra 
from ._linear_algebra import *

from . import _statistics
from ._statistics import *

__all__ += _device.__all__
__all__ += _array_obj.__all__
__all__ += _algorithms.__all__
__all__ += _logic.__all__
__all__ += _math.__all__
__all__ += _construction.__all__
__all__ += _extract_transform.__all__
__all__ += _linear_algebra.__all__
__all__ += _statistics.__all__
__all__ += ["__library_dir__", "__af_version__"]