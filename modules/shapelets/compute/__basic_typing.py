from __future__ import annotations
from typing import Sequence, Union, Tuple
import numpy.typing as npt
from . import _pygauss

_BoolLike = npt._BoolLike
_IntLike = npt._IntLike
_FloatLike = npt._FloatLike
_ComplexLike = npt._ComplexLike
_NumberLike = npt._NumberLike
_ScalarLike = npt._ScalarLike

ArrayLike = npt.ArrayLike
DataTypeLike = npt.DTypeLike
Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]

class Backend:
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...
    CPU: _pygauss.Backend # value = <Backend.CPU: 1>
    CUDA: _pygauss.Backend # value = <Backend.CUDA: 2>
    Default: _pygauss.Backend # value = <Backend.Default: 0>
    OpenCL: _pygauss.Backend # value = <Backend.OpenCL: 4>
    __members__: dict # value = {'Default': <Backend.Default: 0>, 'CPU': <Backend.CPU: 1>, 'CUDA': <Backend.CUDA: 2>, 'OpenCL': <Backend.OpenCL: 4>}

