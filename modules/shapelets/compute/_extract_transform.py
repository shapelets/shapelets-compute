from __future__ import annotations
from typing import List, overload, Literal

from .__basic_typing import ArrayLike, Shape, DataTypeLike
from ._array_obj import ShapeletsArray
from . import _pygauss

BorderType = Literal['clampedge','periodic','symmetric','zero']

def __pygauss_fill_type(type: BorderType):
    if type=='clampedge':
        return _pygauss.BorderType.ClampEdge
    elif type=='periodic':
        return _pygauss.BorderType.Periodic
    elif type=='symmetric':
        return _pygauss.BorderType.Symmetric
    elif type=='zero':
        return _pygauss.BorderType.Zero 
    else:
        raise ValueError("Unknown border type")   

def cast(array_like: ArrayLike, dtype: DataTypeLike) -> ShapeletsArray:
    """
    """
    return _pygauss.cast(array_like, dtype)

def flat(array_like: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.flat(array_like)

def flip(array_like: ArrayLike, dimension: int = 0) -> ShapeletsArray:
    """
    """
    return _pygauss.flip(array_like, dimension)

def join(lst: List[ArrayLike], dimension: int = 0) -> ShapeletsArray: 
    """
    """
    return _pygauss.join(lst, dimension)

def lower(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray: 
    """
    """
    return _pygauss.lower(array_like, unit_diag)

def pad(array_like: ArrayLike, begin: Shape, end: Shape, fill_type: BorderType) -> ShapeletsArray: 
    """
    """
    return _pygauss.pad(array_like, begin, end, __pygauss_fill_type(fill_type))

def reorder(array_like: ArrayLike, x: int, y: int = 1, z: int = 2, w: int = 3) -> ShapeletsArray: 
    """
    """
    return _pygauss.reorder(array_like, x, y, z, w)

def reshape(array_like: ArrayLike, shape: Shape) -> ShapeletsArray: 
    """
    """
    return _pygauss.reshape(array_like, shape)

def shift(array_like: ArrayLike, x: int, y: int = 0, z: int = 0, w: int = 0) -> ShapeletsArray: 
    """
    """
    return _pygauss.shift(array_like, x, y, z, w)

def tile(array_like: ArrayLike, x: int, y: int = 1, z: int = 1, w: int = 1) -> ShapeletsArray:      
    """
    """
    return _pygauss.tile(array_like, x, y, z, w)

def transpose(array_like: ArrayLike, dims: bool = False) -> ShapeletsArray:      
    """
    """
    return _pygauss.transpose(array_like, dims)

def upper(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray:      
    """
    """
    return _pygauss.upper(array_like, unit_diag)

def where(condition: ArrayLike, x: ArrayLike = None, y: ArrayLike = None) -> ShapeletsArray:      
    """
    """
    return _pygauss.where(condition, x, y)

def unpack(a: ArrayLike, wx: int, wy: int, sx: int, sy:int, px: int =0, py: int = 0, is_column: bool = True):     
    """
    """
    return _pygauss.unpack(a,wx,wy,sx,sy,px,py,is_column)

def pack(a: ArrayLike, ox: int, oy: int, wx: int, wy: int, sx: int, sy: int, px: int =0, py: int = 0, is_column: bool = True):     
    """
    """
    return _pygauss.pack(a, ox, oy, wx, wy, sx, sy, px, py, is_column)


__all__ = [
    "pad", "lower", "upper", "reshape", "flat", "flip", "reorder",
    "shift", "tile", "transpose", "cast", "join", "where",
    "pack", "unpack",
    "BorderType"
]

