from __future__ import annotations

from typing import Union
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
from . import _pygauss

def visvalingam(x: ArrayLike, y: ArrayLike, num_points: int) -> ShapeletsArray: 
    """
    Reduces a set of points by applying the Visvalingam method (minimum triangle area) until the number
    of points is reduced to numPoints.

    This method will return real points of the series, that is, no interpolation or calculated values 
    will be added to the returned sequence.

    Parameters
    ----------
    x: ArrayLike
        Column vector (nx1) representing x axis values 
    
    y: ArrayLike
        Column vector (nx1) representing y axis values
    
    num_points: int
        Number of points to reduce

    Returns
    -------
    ShapeletsArray
        An array of shape nx2, where the columns are the x and y axis of those points of the original series 
        that should be kept to maximize the fidelity of the original series.

    References
    ----------
    [1] Line generalisation by repeated elimination of points
        M. Visvalingam and J. D. Whyatt. The Cartographic Journal, 1993.
    """
    return _pygauss.visvalingam(x, y, num_points)


def pip(x: ArrayLike, y: ArrayLike, ips: int) -> ShapeletsArray:    
    """
    Perceptually Important Points

    Calculates the location of Perceptually Important Points (PIP) in the sequence. 

    Parameters
    ----------
    x: ArrayLike
        Column vector (nx1) representing x axis values 

    y: ArrayLike
        Column vector (nx1) representing y axis values

    num_points: int
        Number of points to reduce
    
    References
    ----------
    [1] Representing financial time series based on data point importance. 
        Fu TC, Chung FL, Luk R, and Ng CM. Engineering Applications of Artificial Intelligence, 21(2):277-300, 2008.
    """
    return _pygauss.pip(x, y, ips)

def paa(x: ArrayLike, y: ArrayLike, bins: int) -> ShapeletsArray:        
    r"""
    Piecewise Aggregate Approximation (PAA)

    .. math::

        \bar{x}_{i} = \frac{M}{n} \sum_{j=n/M(i-1)+1}^{(n/M)i} x_{j}.

    """
    return _pygauss.paa(x, y, bins)


__all__ = [
    "visvalingam", "pip", "paa"
]


