from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

def decimal_scaling(tss: ArrayLike) -> ShapeletsArray: 
    r"""
    Normalizes sequences (column vectors) using decimal scaling

    Decimal scaling is a transformation whereby each value of the sequence is adjusted by 

    .. math::
        a' = \frac{a}{10^d}

    where ``d`` is the smallest integer such that :math:`max(\left | a' \right |) < 1`
    
    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.decimal_scaling(a)
    >>> anorm.T
    [1 5 1 1]
        0.0100    -0.2000     0.1000     0.0200     0.0100 
    """
    return _pygauss.decimal_scaling(tss)

def minmax_norm(tss: ArrayLike, high: float = 1.0, low: float = 0.0) -> ShapeletsArray: 
    """
    Normalizes sequences (column vectors) using min-max algorithm

    This algorithm maps maximum and minimun values in the sequences to an arbitrary new high and 
    new low.  The rest of the values are reduced proportionally.
    
    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    high: float (default: 1.0)
        New high value
    
    low: float (default: 1.0)
        New low value

    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.minmax_norm(a)
    >>> anorm.T
    [1 5 1 1]
        0.7000     0.0000     1.0000     0.7333     0.7000
    """
    return _pygauss.minmax_norm(tss, high, low)

def mean_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Applies mean normalization to input sequences (column vectors)

    Mean normalization simply substracts the mean of the sequence to each value and 
    scales by the difference between the minimum and maximum values.

    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.mean_norm(a)    
    >>> anorm.T
    [1 5 1 1]
        0.0733    -0.6267     0.3733     0.1067     0.0733    
    """
    return _pygauss.mean_norm(array_like)

def zscore(tss: ArrayLike, axis: int = 0, ddof: int = 0) -> ShapeletsArray: 
    """
    Applies z-normalization to input sequences

    z-normalization simply substracts the mean of the sequence to each value and 
    scales by the standard deviation.

    Parameters
    ----------
    tss: ArrayLike
        Input array expression
    
    axis: int (defaul: 0)
        Axis where the sequences are defined.

    ddof: int (default: 0)
        Degress of freedom for the standard deviation 

    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences
    
    See also
    --------
    std

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.zscore(a)    
    >>> anorm.T
    [1 5 1 1]
        0.2203    -1.8823     1.1213     0.3204     0.2203   
    """
    return _pygauss.zscore(tss, axis, ddof)

def unit_length_norm(tss: ArrayLike) -> ShapeletsArray: 
    """
    Normalizes using unit length strategy.

    This normalization scales each element of the sequence by its norm.

    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.unit_length_norm(a)    
    >>> anorm.T
    [1 5 1 1]
        0.0445    -0.8891     0.4446     0.0889     0.0445 
    >>> sc.norm(anorm)
    0.9999999999999999

    """
    return _pygauss.unit_length_norm(tss)

def median_norm(tss: ArrayLike) -> ShapeletsArray: 
    """
    Normalizes using median as a scaling factor.

    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.median_norm(a)    
    >>> anorm.T
    [1 5 1 1]
        1.0000   -20.0000    10.0000     2.0000     1.0000     
    >>> sc.statistics.median(a)
    1.0
    """
    return _pygauss.median_norm(tss)

def logistic_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Normalization through the logistic function

    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.logistic_norm(a)    
    >>> anorm.T
    [1 5 1 1]
        0.7311     0.0000     1.0000     0.8808     0.7311
    """
    return _pygauss.logistic_norm(array_like)

def tanh_norm(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Normalization through the hyperbolic tangent.

    Parameters
    ----------
    tss: ArrayLike
        Columnar sequences.
    
    Results
    -------
    ShapeletsArray
        A normalized version of the input sequences

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1.0, -20.0, 10.0, 2.0, 1.0])
    >>> anorm = sc.normalization.tanh_norm(a)    
    >>> anorm.T
    [1 5 1 1]
        0.7616    -1.0000     1.0000     0.9640     0.7616
    """
    return _pygauss.tanh_norm(array_like)

__all__ = [ 
    "tanh_norm", "logistic_norm", "median_norm", "unit_length_norm", 
    "zscore", "mean_norm", "minmax_norm", "decimal_scaling"
]