import os

# os.environ["AF_JIT_KERNEL_TRACE"] = "stderr"
# os.environ["AF_TRACE"]="all"
# os.environ["AF_SHOW_LOAD_PATH"]="1"
# os.environ["AF_PRINT_ERRORS"]="1"


# Pro
library_dir = None 

# current location
compute_dir = compute_dir = os.path.abspath(os.path.dirname(__file__))

# look if the production directory exists...
if os.name == 'darwin':
    library_dir = os.path.join(compute_dir, '..', '.dylib')
else:
    library_dir = os.path.join(compute_dir, '..', '.libs')

# In development, fall back to external / arrayfire / lib folder
if not os.path.exists(library_dir):
    if os.name == "posix":
        library_dir = os.path.join(compute_dir, '..', '..', '..', 'external', 'arrayfire', 'lib64')
    else:
        library_dir = os.path.join(compute_dir, '..', '..', '..', 'external', 'arrayfire', 'lib')

if not os.path.exists(library_dir):
    raise RuntimeError("No valid location can be stablished for native libraries.")

# Let AF know where the RT libraries are
os.environ["AF_BUILD_LIB_CUSTOM_PATH"] = library_dir

if os.name == 'nt':
    import warnings 
    import glob
    try:
        from ctypes import PyDLL
        # Ensure present folder is present in the path
        os.environ['PATH'] = library_dir + ';' + compute_dir + ';' + os.environ['PATH']
    except:
        warnings.warn('Unable to perform initialization', stacklevel=2)    
    else:
        filenames = []
        # For some reason, we are forced to load the library using 
        # the semantics of LoadLibrary and not LoadLibraryEx in 
        # windows system.  It requires more investigation but it may 
        # have something to do with the fact we are loading arrayfire 
        for filename in glob.glob(os.path.join(compute_dir, '_pygauss*.pyd')):  
            PyDLL(filename, winmode= 0)  
            filenames.append(filename)

        if len(filenames) > 1:
            warnings.warn("Loaded more than one pyd library in compute folder: %s" % repr(filenames), stacklevel=2)

del library_dir
del compute_dir

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