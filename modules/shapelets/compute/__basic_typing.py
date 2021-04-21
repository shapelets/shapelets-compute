from __future__ import annotations
from typing import Sequence, Union, Tuple
import numpy.typing as npt

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
