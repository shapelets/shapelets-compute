import sys

try:
    __SHAPELETS_SETUP__
except NameError:
    __SHAPELETS_SETUP__ = False

if __SHAPELETS_SETUP__:
    sys.stderr.write("Running from source directory.\n")
else:
    # Normal initialization here
    #
    # Get all the functions and modules from pygauss
    # which have been designed to be imported this way!
    from .pygauss import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
