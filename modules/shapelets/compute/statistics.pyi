from __future__ import annotations

from typing import Union, Optional, Tuple
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
from . import _pygauss

FloatOrComplex = Union[complex, float]

class XCorrScale:
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str:...
    NoScale: _pygauss.XCorrScale
    Biased: _pygauss.XCorrScale
    Unbiased: _pygauss.XCorrScale
    Coeff: _pygauss.XCorrScale
    __members__: dict 

def mean(data: ArrayLike, weights: ArrayLike = None, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    
    TODO
    """

def median(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]:
    """
    TODO
    """

def var(data: ArrayLike, weigths: ArrayLike = None, dim: int = None, biased: bool = False) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    TODO
    """

def std(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    TODO
    """

def skewness(data: ArrayLike) -> ShapeletsArray: 
    """
    Calculates the sample skewness of data, calculated with the adjusted Fisher-Pearson standardized moment
    coefficient G1.

    The input data parameter may contain as many sequences as required, computing the skewness in parallel.

    Parameters
    ----------
    data: ArrayLike
    A NxM two dimensional matrix, where there are M sequences of N points.

    Returns
    -------
    The skewness for each series, represented as a 1xM vector.
    """

def kurtosis(data: ArrayLike) -> ShapeletsArray: 
    """
    Calculates the sample kurtosis of data, calculated with the adjusted Fisher-Pearson standardized moment
    coefficient G2.

    The input data parameter may contain as many sequences as required, computing the kurtosis in parallel.

    Parameters
    ----------
    data: ArrayLike
    A NxM two dimensional matrix, where there are M sequences of N points.

    Returns
    -------
    The kurtosis for each series, represented as a 1xM vector.
    """

def moment(data: ArrayLike, k: int) -> ShapeletsArray: 
    """
    Computes the kth moment of data.  

    The input data parameter may contain as many sequences as required, computing all moments in parallel.
    
    Parameters
    ----------
    data: ArrayLike
    A NxM two dimensional matrix, where there are M sequences of N points.
    k: int 
    The moment to compute.

    Returns
    -------
    The kth moment for each series, represented as a 1xM vector.
    """

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
    Covariance matrix of all the series in data.
    """

def corrcoef(data: ArrayLike, ddof: int = 1) -> ShapeletsArray:
    """
    TODO
    """
    
def xcov(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = XCorrScale.NoScale) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    TODO
    """

def xcorr(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = XCorrScale.NoScale) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    TODO
    """

def acorr(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = XCorrScale.NoScale) -> ShapeletsArray: 
    """
    TODO
    """

def acov(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = XCorrScale.NoScale) -> ShapeletsArray: 
    """
    TODO
    """

def topk_max(data: ArrayLike, k: int) -> ShapeletsArray: 
    """
    TODO
    """

def topk_min(data: ArrayLike, k: int) -> ShapeletsArray: 
    """
    TODO
    """


# def ljungbox(data: ArrayLike, lags: int) -> ShapeletsArray: ...
# def quantile(data: ArrayLike, quantiles: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: ...
# def quantiles_cut(data: ArrayLike, regions: int, is_sorted: bool = False) -> ShapeletsArray: ...
