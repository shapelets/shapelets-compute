# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from __future__ import annotations
from typing import Sequence, Union, Tuple, Any, overload
import sys
import numpy as np

DataTypeLike = Union[np.dtype, None, str]

if sys.version_info >= (3, 8):
    from typing import Protocol

    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if HAVE_PROTOCOL:
    class _SupportsArray(Protocol):
        @overload
        def __array__(self, __dtype: DataTypeLike = ...) -> np.ndarray: ...

        @overload
        def __array__(self, dtype: DataTypeLike = ...) -> np.ndarray: ...
else:
    _SupportsArray = Any

_BoolLike = Union[bool, np.bool_]
_IntLike = Union[int, np.integer]
_FloatLike = Union[_IntLike, float, np.floating]
_ComplexLike = Union[_FloatLike, complex, np.complexfloating]
_NumberLike = Union[int, float, complex, np.number, np.bool_]
_ScalarLike = _ScalarLike = Union[int, float, complex, bytes, np.generic]

ArrayLike = Union[_ScalarLike, Sequence[_ScalarLike], Sequence[Sequence[Any]], _SupportsArray]
Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
