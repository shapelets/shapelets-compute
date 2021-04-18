from __future__ import annotations

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

try:
    __SHAPELETS_SETUP__
except NameError:
    __SHAPELETS_SETUP__ = False

from sys import stderr

if __SHAPELETS_SETUP__:
    stderr.write("Running from source directory.\n")
else:
    # Normal initialization here
    from . import compute
    from . import generators
    from . import data
    __all__ = ["compute", "generators", "data"]

del stderr
