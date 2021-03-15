from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    DistanceType,
    pdist as _pdist,
    cdist as _cdist
    
)   

def pdist(tss: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray: 
    """
    Pairwise distances between observations in n-dimensional space.
    """
    return _pdist(tss, metric, **kwargs)

def cdist(xa: ArrayLike, xb: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray: 
    """
    Compute distance between each pair of the two collections of inputs.
    """
    return _cdist(xa, xb, metric, **kwargs)

def euclidian(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the Euclidean distance between two 1-D arrays.
    """
    return cdist(a, b, DistanceType.Euclidian)

def hamming(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the Hamming distance between two 1-D arrays.
    """
    return cdist(a, b, DistanceType.Hamming)

def cityblock(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the City Block (a.k.a. Manhantan) distance between two 1-D arrays.
    """
    return cdist(a, b, DistanceType.Manhattan)

def sbd(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the Shape-Based distance (SBD) between two 1-D arrays.
    
    It computes the normalized cross-correlation and it returns 1.0 minus the value 
    that maximizes the correlation value between each pair of time series.
    """
    return cdist(a, b, DistanceType.SBD)

def dtw(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the Dynamic Time Warping Distance between two 1-D arrays.
    """
    return cdist(a, b, DistanceType.DTW)

def mpdist(a: ArrayLike, b: ArrayLike) -> ShapeletsArray: 
    """
    Computes the Matrix Profile Distance between two 1-D arrays.
    """
    raise ValueError("Not implemented")
    
def chebyshev(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Chebyshev)

def minkowshi(a: ArrayLike, b: ArrayLike, p: float) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Minkowshi, p=p)
