from __future__ import annotations
from typing import Union, Optional, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
from . import _pygauss


FloatOrComplex = Union[complex, float]
XCorrScale = Literal['noscale', 'biased', 'unbiased', 'coeff']

def __convert_xcorr_scale(v: XCorrScale):
    if v == 'noscale':
        return _pygauss.XCorrScale.NoScale
    elif v == 'biased':
        return _pygauss.XCorrScale.Biased
    elif v == 'unbiased':
        return _pygauss.XCorrScale.Unbiased
    elif v == 'coeff':
        return _pygauss.XCorrScale.Coeff
    else:
        raise ValueError("Unknown XCorrScale")

def mean(data: ArrayLike, weights: ArrayLike = None, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    TODO
    """
    return _pygauss.mean(data, weights, dim)

def median(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]:
    """
    TODO
    """
    return _pygauss.median(data, dim)

def std(data: ArrayLike, ddof: int = 1, dim: int = 0) -> ShapeletsArray:
    """
    """
    return _pygauss.std(data, ddof, dim)

def var(data: ArrayLike, ddof: int = 1, dim: int = 0) -> ShapeletsArray:
    """
    """
    return _pygauss.var(data, ddof, dim)

def moment(data: ArrayLike, k: int, dim: int = 0) -> ShapeletsArray: 
    """
    Computes the kth statistical moment of data.  

    Parameters
    ----------
    data: ArrayLike
        Input data
    
    k: int 
        The moment to compute.
    
    dim: int, defaults to 0
        Dimension to iterate through 

    Returns
    -------
    ShapeletsArray
        The kth moment of each sub-vector found by iterating through a particular dimension.
    """
    return _pygauss.moment(data, k, dim)

def kurtosis(data: ArrayLike, dim: int = 0) -> ShapeletsArray: 
    """
    Calculates the sample kurtosis of data, calculated with the adjusted Fisher-Pearson standardized moment
    coefficient G2.

    The input data parameter may contain as many sequences as required, computing the kurtosis in parallel.

    Parameters
    ----------
    data: ArrayLike
        Input Data
    
    dim: int.  Defaults to 0
        Dimension to iterate through 
    
    Returns
    -------
    ShapeletsArray
        The kurtosis of each sub-vector found by iterating through a particular dimension.
    """
    return _pygauss.kurtosis(data, dim)


def skewness(data: ArrayLike, dim: int = 0) -> ShapeletsArray: 
    """
    Calculates the sample skewness of data, calculated with the adjusted Fisher-Pearson standardized moment
    coefficient G1.

    The input data parameter may contain as many sequences as required, computing the skewness in parallel.

    Parameters
    ----------
    data: ArrayLike
        Input Data
    dim: int.  Defaults to 0
        Dimension to iterate through 
    
    Returns
    -------
    ShapeletsArray
        The skewness for each sub-vector found by iterating through a particular dimension.

    """
    return _pygauss.skewness(data, dim)

def cov(data: ArrayLike, ddof: int = 1) -> ShapeletsArray:
    """
    Returns the covariance matrix of the series contained in data.

    Parameters
    ----------
    data: ArrayLike
        A NxM two dimensional matrix, where there are M sequences of N points.
    ddof: bool 

    Returns
    -------
    ShapeletsArray
        Covariance matrix of all the series in data.
    """
    return _pygauss.cov(data, ddof)

def corrcoef(data: ArrayLike, ddof: int = 1) -> ShapeletsArray:
    """
    TODO
    """
    return _pygauss.corrcoef(data, ddof)
    
class XResults(NamedTuple):
    lags: ShapeletsArray
    values: ShapeletsArray

def xcov(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> XResults: 
    """
    TODO
    """    
    return XResults(*_pygauss.xcov(xss, yss, maxlag, __convert_xcorr_scale(scale)))

def xcorr(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> XResults: 
    """
    TODO
    """    
    return XResults(*_pygauss.xcorr(xss, yss, maxlag, __convert_xcorr_scale(scale)))

def acorr(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.acorr(data, maxlag, __convert_xcorr_scale(scale))

def acov(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> ShapeletsArray: 
    """
    TODO
    """
    return _pygauss.acov(data, maxlag, __convert_xcorr_scale(scale))

class TopKResult(NamedTuple):
    values: ShapeletsArray
    """Top values"""
    indices: ShapeletsArray
    """Indices of those values"""

def topk_max(data: ArrayLike, k: int) -> TopKResult: 
    """
    TODO
    """
    return TopKResult(*_pygauss.topk_max(data, k))

def topk_min(data: ArrayLike, k: int) -> TopKResult: 
    """
    TODO
    """    
    return TopKResult(*_pygauss.topk_min(data, k))
