from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    decimal_scaling as __decimal_scaling, minmax_norm as __minmax_norm, 
    mean_norm as __mean_norm, zscore as __zscore, 
    unit_length_norm as __unit_length_norm,  median_norm as __median_norm, 
    logistic_norm as __logistic_norm, tanh_norm as __tanh_norm
)

def decimal_scaling(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __decimal_scaling(array_like)

def minmax_norm(array_like: ArrayLike, high: float = 1.0, low: float = 0.0) -> ShapeletsArray: 
    """
    TODO
    """
    return __minmax_norm(array_like, high, low)

def mean_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __mean_norm(array_like)

def zscore(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __zscore(array_like)

def unit_length_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __unit_length_norm(array_like)

def median_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __median_norm(array_like)

def logistic_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __logistic_norm(array_like)

def tanh_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return __tanh_norm(array_like)


__all__ = [
    "decimal_scaling", "minmax_norm", "mean_norm",
    "zscore", "unit_length_norm", "median_norm",
    "logistic_norm", "tanh_norm"
]
