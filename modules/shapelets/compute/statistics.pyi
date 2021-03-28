from typing import Union
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

FloatOrComplex = Union[complex, float]

def mean(data: ArrayLike, weights: ArrayLike = None, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: ... 
def median(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: ...
def var(data: ArrayLike, weigths: ArrayLike = None, dim: int = None, biased: bool = False) -> Union[FloatOrComplex, ShapeletsArray]: ... 
def std(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]: ... 

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

def covariance(data: ArrayLike, unbiased: bool = False) -> ShapeletsArray:
    """
    Returns the covariance matrix of the series contained in data.

    Parameters
    ----------
    data: ArrayLike
    A NxM two dimensional matrix, where there are M sequences of N points.
    biased: bool 
    Determines if the computation adjusts n to `n-1` (false) or not (true).

    Returns
    -------
    Covariance matrix of all the series in data.
    """

def correlation(data: ArrayLike, unbiased: bool = False) -> ShapeletsArray: ...

def cross_covariance(xss: ArrayLike, yss: ArrayLike, unbiased: bool = False) -> ShapeletsArray: ...
def cross_correlation(xss: ArrayLike, yss: ArrayLike, unbiased: bool = False) -> ShapeletsArray: ...

def auto_covariance(data: ArrayLike, unbiased: bool = False) -> ShapeletsArray: ...          
def auto_correlation(data: ArrayLike, max_lag: int, unbiased: bool = False) -> ShapeletsArray: ...

def partial_auto_correlation(data: ArrayLike, lags: ArrayLike) -> ShapeletsArray: ...


def ljungbox(data: ArrayLike, lags: int) -> ShapeletsArray: ...

def quantile(data: ArrayLike, quantiles: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: ...
def quantiles_cut(data: ArrayLike, regions: int, is_sorted: bool = False) -> ShapeletsArray: ...


def topk_max(data: ArrayLike, k: int) -> ShapeletsArray: ...
def topk_min(data: ArrayLike, k: int) -> ShapeletsArray: ...
