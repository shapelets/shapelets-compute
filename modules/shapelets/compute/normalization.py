from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

def detrend(array_like: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.detrend(array_like)

def decimal_scaling(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.decimal_scaling(array_like)

def minmax_norm(array_like: ArrayLike, high: float = 1.0, low: float = 0.0) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.minmax_norm(array_like, high, low)

def mean_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.mean_norm(array_like)

def zscore(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.zscore(array_like)

def unit_length_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.unit_length_norm(array_like)

def median_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.median_norm(array_like)

def logistic_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.logistic_norm(array_like)

def tanh_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.tanh_norm(array_like)

