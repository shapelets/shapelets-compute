# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

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
    from . import worker

    from ._cli import *
    from . import _cli

    __all__ = ["compute", "generators", "data", "worker"]
    __all__ += _cli.__all__

    backends = compute.get_available_backends()
    if len(backends) <= 1:
        import warnings

        if len(backends) == 0:
            msg = """
                No backends available.  Please use shapelets command line tool to 
                install a new backend.  For example: shapelets install cpu
                """
        elif backends[0] == 'cpu':
            msg = "Only one compute device found: " + repr(backends)
            msg += """
                Most of the operations won't be accelerated since the only device found is CPU.  Consider 
                adding OpenCL or CUDA support to your environment to benefit from the accelerated versions of the 
                algorithms this library provides.
                """
        warnings.warn(msg, RuntimeWarning)
    del backends

del stderr
