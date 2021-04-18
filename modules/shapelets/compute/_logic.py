from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
from . import _pygauss

def iscomplex(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.iscomplex(array_like)

def isfinite(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.isfinite(array_like)

def isinf(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.isinf(array_like)

def isnan(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.isnan(array_like)

def isreal(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.isreal(array_like)

def logical_and(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.logical_and(left, right)

def logical_not(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.logical_not(array_like)

def logical_or(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.logical_or(left, right)

def not_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.not_equal(left, right)

def equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.equal(left, right)

def greater(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.greater(left, right)

def greater_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.greater_equal(left, right)

def less(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.less(left, right)

def less_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.less_equal(left, right)


__all__ = [
    "isfinite", "isinf", "isnan", "iscomplex", "isreal", "logical_and", "logical_or", 
    "logical_not", "equal", "not_equal", "greater", "greater_equal", "less", "less_equal"
]