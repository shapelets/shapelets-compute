from typing import Union
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray


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
    An array of shape nx2, where the columns are the x and y axis of those points of the original series 
    that should be kept to maximize the fidelity of the original series.

    References
    ----------
    [1] M. Visvalingam and J. D. Whyatt, Line generalisation by repeated elimination of points,
    The Cartographic Journal, 1993.
    """


# def sax(data: ArrayLike, alphabet_size: int) -> ShapeletsArray: 
#     """
#     Symbolic Aggregate approXimation (SAX). 
    
#     It transforms a numeric time series into a time series of symbols with the same size. The algorithm 
#     was proposed by Lin et al., and extends the PAA-based approach inheriting the original algorithm 
#     simplicity and low computational complexity while providing satisfactory sensitivity and selectivity in
#     range query processing. Moreover, the use of a symbolic representation opened a door to the existing 
#     wealth of data-structures and string-manipulation algorithms in computer science such as hashing, 
#     regular expression, pattern matching, suffix trees, and grammatical inference.

#     Parameters
#     ----------
#     data: ArrayLike
#     Columnar vector nx1 containing the data
#     alphabe_size: int

#     Returns
#     -------

#     References
#     ----------
#     [1] Lin, J., Keogh, E., Lonardi, S. & Chiu, B. (2003) A Symbolic Representation of Time Series, with 
#     Implications for Streaming Algorithms. In proceedings of the 8th ACM SIGMOD Workshop on Research 
#     Issues in Data Mining and Knowledge Discovery. San Diego, CA. June 13.
#     """
    
def pip(x: ArrayLike, y: ArrayLike, ips: int) -> ShapeletsArray:    
    """
    Perceptually Important Points

    Calculates the location of Perceptually Important Points (PIP) in the sequence. 

    Parameters
    ----------

    
    References
    ----------
    Fu TC, Chung FL, Luk R, and Ng CM. Representing financial time series based on data point importance. 
    Engineering Applications of Artificial Intelligence, 21(2):277-300, 2008.
    """

def paa(x: ArrayLike, y: ArrayLike, bins: int) -> ShapeletsArray:        
    """
    Piecewise Aggregate Approximation (PAA)

    ..math::

        \\bar{x}_{i} = \\frac{M}{n} \\sum_{j=n/M(i-1)+1}^{(n/M)i} x_{j}.


    """


