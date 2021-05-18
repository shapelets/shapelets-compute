from __future__ import annotations
from typing import Sequence, Union, Tuple, Any, overload
import sys
import numpy as np

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
        def __array__(self, __dtype: DTypeLike = ...) -> np.ndarray: ...
        @overload
        def __array__(self, dtype: DTypeLike = ...) -> np.ndarray: ...
else:
    _SupportsArray = Any

_BoolLike = Union[bool, np.bool_]
_IntLike = Union[int, np.integer]
_FloatLike = Union[_IntLike, float, np.floating]
_ComplexLike = Union[_FloatLike, complex, np.complexfloating]
_NumberLike = Union[int, float, complex, np.number, np.bool_]
_ScalarLike = _ScalarLike = Union[int, float, complex, bytes, np.generic]

ArrayLike = Union[_ScalarLike, Sequence[_ScalarLike], Sequence[Sequence[Any]], _SupportsArray]
DTypeLike = Union[np.dtype, None, str]
Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]

