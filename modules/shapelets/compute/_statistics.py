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
    Computes the arithmetic mean along an (optional) dimension

    Parameters
    ----------
    data: ArrayLike
        Input array expression
    
    weights: Optional ArrayLike (default: None)
        An array of weights associated to each of the values in ``data``.  This parameter, when specified, must be of 
        the same dimensions as ``data``.
    
    dim: Optional int (default: None)
        Dimension for the operation.  When no dimension is given, a scalar value will be returned.

    Results
    -------
    FloatOrComplex
        When no dimension is given or, alternatively,
    ShapeletsArray
        When the operation occurs over a concrete dimension.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.random.randn((10,2))
    >>> w = sc.random.randn((10,2))
    >>> sc.statistics.mean(a)
    0.31603485345840454
    >>> sc.statistics.mean(a, dim=0)
    [1 2 1 1]
        0.0413     0.5907 
    >>> sc.statistics.mean(a, w)
    1.7622544765472412
    >>> sc.statistics.mean(a, w, 0)
    [1 2 1 1]
        -0.1348     2.6414     
    """
    return _pygauss.mean(data, weights, dim)

def median(data: ArrayLike, dim: int = None) -> Union[FloatOrComplex, ShapeletsArray]:
    """
    Computes the median over a (optional) dimension

    Parameters
    ----------
    data: ArrayLike
        Input array expression
    
    dim: Optional int (default: None)
        Dimension for the operation.  When no dimension is given, a scalar value will be returned.

    Results
    -------
    FloatOrComplex
        When no dimension is given or, alternatively,

    ShapeletsArray
        When the operation occurs over a concrete dimension.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,1,1,3,1,1], "float32")
    >>> sc.statistics.median(a)
    1.0
    """
    return _pygauss.median(data, dim)

def std(data: ArrayLike, ddof: int = 1, dim: int = 0) -> ShapeletsArray:
    """
    Computes the standard deviation across a particular dimension.

    Parameters
    ----------
    data: ArrayLike
        Input array expression
    
    ddof: int (default: 1)
        Degrees of freedom for the calculation.  In standard statistical practice, the 
        default value provides an unbiased estimator of the variance of the infinite population. 
        Setting this parameter to 0, it provides a maximum likelihood estimate of the variance 
        for normally distributed variables.  

    dim: int (default: 0)
        Dimension for the operation. 
    
    Returns
    -------
    ShapeletsArray
        New array with the results of the computation.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,1,1,3,1,1.], [1,2,3,4,6,7]]).T
    >>> sc.statistics.std(a, 0)
    [1 2 1 1]
        0.7454     2.1148     
    >>> sc.statistics.std(a, 1)
    [1 2 1 1]
        0.8165     2.3166     
    """
    return _pygauss.std(data, ddof, dim)

def var(data: ArrayLike, ddof: int = 1, dim: int = 0) -> ShapeletsArray:
    """
    Computes the variance across a particular dimension.

    Parameters
    ----------
    data: ArrayLike
        Input array expression
    
    ddof: int (default: 1)
        Degrees of freedom for the calculation.  In standard statistical practice, the 
        default value provides an unbiased estimator of the variance of the infinite population. 
        Setting this parameter to 0, it provides a maximum likelihood estimate of the variance 
        for normally distributed variables.  

    dim: int (default: 0)
        Dimension for the operation. 
    
    Returns
    -------
    ShapeletsArray
        New array with the results of the computation.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,1,1,1,1,1.], [1,2,3,4,6,7]]).T
    >>> sc.statistics.var(a, 0)
    [1 2 1 1]
        0.0000     4.4722     
    >>> sc.statistics.var(a, 1)
    [1 2 1 1]
        0.0000     5.3667   
    """
    return _pygauss.var(data, ddof, dim)

def moment(data: ArrayLike, k: int, dim: int = 0) -> ShapeletsArray: 
    r"""
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
    r"""
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
    r"""
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
    r"""
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
    r"""
    Return Pearson product-moment correlation coefficients.

    .. attention::
       Please note this implementation treats columns as variables whilst
       the default behaviour of numpy is to treat rows as variables.  See 
       examples below.

    Parameters
    ----------
    data: ArrayLike
        NxM matrix with M variables and N observations.
    
    ddof: int (default: 1)
        Degrees of freedom

    Returns
    -------
    ShapeletsArray
        The correlation coefficient matrix of the variables

    Examples
    --------
    >>> import numpy as np
    >>> import shapelets.compute as sc
    >>> data = [[1., 2, 3, 4], [2., 3, 4, 5], [3., 4, 5, 6], [7., 8, 9, 0]]
    >>> np.corrcoef(data, rowvar=False)
    array([[ 1.        ,  1.        ,  1.        , -0.80722892],
           [ 1.        ,  1.        ,  1.        , -0.80722892],
           [ 1.        ,  1.        ,  1.        , -0.80722892],
           [-0.80722892, -0.80722892, -0.80722892,  1.        ]])
    >>> sc.statistics.corrcoef(data)
    [4 4 1 1]
        1.0000     1.0000     1.0000    -0.8072 
        1.0000     1.0000     1.0000    -0.8072 
        1.0000     1.0000     1.0000    -0.8072 
       -0.8072    -0.8072    -0.8072     1.0000 
    """
    return _pygauss.corrcoef(data, ddof)
    
class XCoResults(NamedTuple):
    lags: ShapeletsArray
    values: ShapeletsArray

def xcov(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> XCoResults: 
    r"""
    Estimates the cross-covariance of each pair of column vectors in ``xss`` and ``yss``. 

    Parameters
    ----------
    xss: ArrayLike
        Input matrix nxN, with N column vectors of size n

    yss: ArrayLike
        Input matrix mxM, with M column vectors of size m

    maxlag: Optional int (default: None)
        Max lag to compute.  If not specified, it will default to ``max(n,m)-1``

    scale: Optional XCorrScale (default: 'noscale')
        Scales the results as per the following preferences:

        - 'noscale': the computation will be returned without further adjustments
        - 'biased': returns the biased average
        - 'unbiased': returns the unbiased average
        - 'coeff': raw results are scaled by :math:`\frac{1}{rms(xss)^Trms(yss)}`, where
        `rms` stands for ``root mean squared`` of the column vectors on each of the inputs.

    Returns
    -------
    XCoResults
        Named tuple of lags and estimates.  The `lags` field will be an array ranging from 
        `-maxlag .. maxlag`.

    See also
    --------
    xcorr

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1., 2, 3, 4])
    >>> b = sc.array([3., 4, 5, 6])
    >>> lags, values = sc.statistics.xcov(a,b, scale='coeff')
    >>> lags.T
    [1 7 1 1]
        -3.0000    -2.0000    -1.0000     0.0000     1.0000     2.0000     3.0000     
    >>> values.T
    [1 7 1 1]
        -0.4500    -0.3000     0.2500     1.0000     0.2500    -0.3000    -0.4500     

    """    
    return XCoResults(*_pygauss.xcov(xss, yss, maxlag, __convert_xcorr_scale(scale)))

def xcorr(xss: ArrayLike, yss: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> XCoResults: 
    r"""
    Cross-correlation estimation.

    Estimates the cross-correlation of each pair of column vectors in ``xss`` and ``yss``.  
    The cross-correlation between two vectors is defined as :math:`R_{xy}(k) = \sum_{i=1}^{n}x_{i+k} \bar{y}_i`

    Parameters
    ----------
    xss: ArrayLike
        Input matrix nxN, with N column vectors of size n

    yss: ArrayLike
        Input matrix mxM, with M column vectors of size m

    maxlag: Optional int (default: None)
        Max lag to compute.  If not specified, it will default to ``max(n,m)-1``

    scale: Optional XCorrScale (default: 'noscale')
        Scales the results as per the following preferences:

        - 'noscale': the computation will be returned without further adjustments
        - 'biased': returns the biased average
        - 'unbiased': returns the unbiased average
        - 'coeff': raw results are scaled by :math:`\frac{1}{rms(xss)^Trms(yss)}`, where
        `rms` stands for ``root mean squared`` of the column vectors on each of the inputs.

    Returns
    -------
    XCoResults
        Named tuple of lags and estimates.  The `lags` field will be an array ranging from 
        `-maxlag .. maxlag`.
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1., 2, 3, 4])
    >>> b = sc.array([3., 4, 5, 6])
    >>> lags, values = sc.statistics.xcorr(a,b, scale='coeff')
    >>> lags.T
    [1 7 1 1]
        -3.0000    -2.0000    -1.0000     0.0000     1.0000     2.0000     3.0000 
    >>> values.T
    [1 7 1 1]
        0.1181     0.3347     0.6300     0.9844     0.7481     0.4922     0.2362
    """    
    return XCoResults(*_pygauss.xcorr(xss, yss, maxlag, __convert_xcorr_scale(scale)))


def acorr(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> ShapeletsArray: 
    r"""
    Auto-correlation estimation of each column vector.

    Parameters
    ----------
    data: ArrayLike
        Input array, nxN, of N column vectors of size n

    maxlag: Optional int (default: None)
        Max lag to compute.  If not specified, it will default to ``n-1``

    scale: Optional XCorrScale (default: 'noscale')
        Scales the results as per the following preferences:

        - 'noscale': the computation will be returned without further adjustments
        - 'biased': returns the biased average
        - 'unbiased': returns the unbiased average
        - 'coeff': raw results are scaled by :math:`\frac{1}{rms(xss)^Trms(yss)}`, where
        `rms` stands for ``root mean squared`` of the column vectors on each of the inputs.    

    Returns
    -------
    ShapeletsArray
        A new array with the autocorrelation estimation.  The array will have ``2*n-1`` rows and 
        as many columns and data (N).

    See also
    --------
    acov

    Example
    -------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1., 2, 3, 4])
    >>> sc.statistics.acorr(a)
    [7 1 1 1]
         4.0000 
        11.0000 
        20.0000 
        30.0000 
        20.0000 
        11.0000 
         4.0000     
    """
    return _pygauss.acorr(data, maxlag, __convert_xcorr_scale(scale))

def acov(data: ArrayLike, maxlag: Optional[int] = None, scale: Optional[XCorrScale] = 'noscale') -> ShapeletsArray: 
    r"""
    Auto-covariance estimation.

    Parameters
    ----------
    data: ArrayLike
        Input array, nxN, of N column vectors of size n

    maxlag: Optional int (default: None)
        Max lag to compute.  If not specified, it will default to ``n-1``

    scale: Optional XCorrScale (default: 'noscale')
        Scales the results as per the following preferences:

        - 'noscale': the computation will be returned without further adjustments
        - 'biased': returns the biased average
        - 'unbiased': returns the unbiased average
        - 'coeff': raw results are scaled by :math:`\frac{1}{rms(xss)^Trms(yss)}`, where
        `rms` stands for ``root mean squared`` of the column vectors on each of the inputs.    

    Returns
    -------
    ShapeletsArray
        A new array with the autocovariance estimation.  The array will have ``2*n-1`` rows and 
        as many columns and data (N).

    See also
    --------
    acorr

    Example
    -------
    >>> import shapelets.compute as sc
    >>> a = sc.array([2., -1, 3, -2])
    >>> sc.statistics.acov(a)
    [7 1 1 1]
       -3.7500 
        7.5000 
      -12.2500 
       17.0000 
      -12.2500 
        7.5000 
       -3.7500     
    """
    return _pygauss.acov(data, maxlag, __convert_xcorr_scale(scale))

class TopKResult(NamedTuple):
    values: ShapeletsArray
    """Top values"""
    indices: ShapeletsArray
    """Indices of those values"""

def topk_max(data: ArrayLike, k: int, dim: int = 0) -> TopKResult: 
    r"""
    Finds values and indices of the top k maximum values.

    If the array contains multiple values of (positive) infinite values or
    nan values, they will be reported as distinct findings. 

    Parameters
    ----------
    data: ArrayLike
        Multidimensional input array
    
    k: int
        Set how many maximum values are to be returned
    
    dim: int (default: 0)
        Changes the dimension.  

    Returns
    -------
    TopKResult
        A named tuple with values and indices of those values.

    Notes
    -----
    This function is optimized for small values of k.  The order of 
    the returned keys may not be in the same order as they appear 
    in the input array.

    Whilst this function provides the ability to change the dimension 
    of the reducction, the underlying implementation only supports
    setting this value to 0. 
    """
    if dim != 0:
        raise ValueError("Dimensions other than zero are not supported")

    return TopKResult(*_pygauss.topk_max(data, k, dim))

def topk_min(data: ArrayLike, k: int, dim: int = 0) -> TopKResult: 
    r"""
    Finds values and indices of the top k minimum values 

    If the array contains multiple values of negative infinite values or
    nan values, they will be reported as distinct findings.

    Parameters
    ----------
    data: ArrayLike
        Multidimensional input array
    
    k: int
        Set how many minimum values are to be returned
    
    dim: int (default: 0)
        Changes the dimension.  

    Returns
    -------
    TopKResult
        A named tuple with values and indices of those values.

    Notes
    -----
    This function is optimized for small values of k.  The order of 
    the returned keys may not be in the same order as they appear 
    in the input array.

    Whilst this function provides the ability to change the dimension 
    of the reducction, the underlying implementation only supports
    setting this value to 0. 
    """    
    if dim != 0:
        raise ValueError("Dimensions other than zero are not supported")

    return TopKResult(*_pygauss.topk_min(data, k, dim))


__all__ = [
    "mean", "median", "std", "var", "moment",
    "kurtosis", "skewness", "cov", "corrcoef", "xcorr", "xcov",
    "acorr", "acov", "topk_max", "topk_min",
    "XCoResults", "TopKResult", "XCorrScale"
]
