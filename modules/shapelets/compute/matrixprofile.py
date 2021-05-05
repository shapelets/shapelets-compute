from __future__ import annotations

from typing import Any, List, NamedTuple, Optional
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss 
from ._pygauss import Snippet

class Snippet:
    @property
    def index(self) -> int: ...
    @property
    def pct(self) -> float: ...
    @property
    def distances(self) -> ShapeletsArray: ...
    @property
    def indices(self) -> ShapeletsArray: ...
    @property
    def pct(self) -> float: ... 
    @property
    def window(self) -> int: ...
    @property 
    def size(self) -> int: ... 
    def __repr__(self) -> str: ...

class MatrixProfile(NamedTuple):
    profile: ShapeletsArray
    """Array of distances"""
    index: ShapeletsArray
    """Array of indices."""
    window: int
    """Size of the window used to create this matrix profile"""

class MatrixProfileLR(NamedTuple):
    left: MatrixProfile 
    """Left direction"""
    right: MatrixProfile
    """Right direction"""

def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray:
    """
    Mueen’s Algorithm for Similarity Search.

    Multiples queries can be run simultaneously against multiple series.

    Parameters
    ----------
    queries : ArrayLike
        Input queries (column wise)
    series: ArrayLike.  
        Input series (column wise)    

    Returns
    -------
    ShapeletsArray
        Returns a 3-D array, with the following structure:
        - 1st dimension corresponds to the index of the subsequence in the time series.
        - 2nd dimension is indexed by the number of queries.
        - 3rd dimension is indexed by the number of series.
    """
    return _pygauss.mass(queries, series)

def matrix_profile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> MatrixProfile:
    """
    Computes matrix profile.

    Parameters
    ----------
    ts : ArrayLike
        Input time series (column wise)
    w : int
        The window size.
    tb: Optional, ArrayLike.  Defaults to None
        Input time series (column wise)                

    Returns
    -------
    MatrixProfile
        A named tuple with computed distances and indices.

    Notes
    -----
    When only one time series is provided, it computes the matrix profile between ``ts`` and 
    itself, for all possible subsequences of length ``m``, filtering trivial matches. On the 
    other hand, when computed against another time series, ``tb``, 




    """
    return MatrixProfile(*_pygauss.matrixprofile(ta, m, tb))

def matrix_profile_lr(ta: ArrayLike, m: int) -> MatrixProfileLR:
    """
    Computes left and right matrix profiles.

    Parameters
    ----------
    ts : ArrayLike
        Input time series (column wise)
    tsb : ArrayLike
        The time series to compare against (column wise)
    w : int
        The window size.
    
    Returns
    -------
    MatrixProfileLR
        A named tuple with two string properties (``left`` and ``right``) of type 
        :obj:`~shapelets.compute.matrixprofile.MatrixProfile`

    See Also
    --------
    matrix_profile
        For standard matrix profile computation

    References
    ----------
    [1] Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining.
        Zhu, Y.; Imamura, M.; Nikovski, D.N.; Keogh, E.
        `TR2017-168 <https://www.merl.com/publications/docs/TR2017-168.pdf>`_ November 2017
        `Alternative Reference <http://www.cs.ucr.edu/~eamonn/chains_ICDM.pdf>`_
    """
    raw_result = _pygauss.matrixprofileLR(ta, m)
    left_value = MatrixProfile(*raw_result[0])
    right_value = MatrixProfile(*raw_result[1])
    return MatrixProfileLR(left_value, right_value)

def mpdist_vect(ts: ArrayLike, tsb: ArrayLike, w: int, threshold: Optional[float] = 0.05) -> ShapeletsArray:
    """
    Computes a vector of MPDist measures.

    Parameters
    ----------
    ts : ArrayLike
        Input time series.
    tsb : ArrayLike
        The time series to compare against.
    w : int
        The window size.
    
    Returns
    -------
    ShapeletsArray
        Array with with MPDist meassures.

    References
    ----------
    [1] Matrix Proﬁle XII: MPdist: A Novel Time Series Distance Measure to Allow Data Mining in More Challenging Scenarios. 
    Shaghayegh Gharghabi, Shima Imani, Anthony Bagnall, Amirali Darvishzadeh, Eamonn Keogh. 
    ICDM 2018
    DOI: `10.1109/ICDM.2018.00119 <https://doi.org/10.1109/ICDM.2018.00119>`_
    `Alternative Reference <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`_
    """
    return _pygauss.mpdist_vect(ts, tsb, w, threshold)

def cac(profile: ArrayLike, index: ArrayLike, window: int) -> ShapeletsArray:
    """
    Returns the implied ``Corrected Arc Crossings`` curve.

    Parameters
    ----------
    profile: ArrayLike
        The array of distances obtained from a matrix profile analysis.
    
    index: ArrayLike
        The array of indices obtained from a matrix profile analysis.

    window: int
        The size of the window applied during the matrix profile analysis.

    Returns
    -------
    ShapeletsArray
        Implied corrected arc crossings.

    See Also
    --------
    segment
        Returns a list of indices where changes in regime are observable from the 
        profile and index arrays.

    References
    ----------
    [1] Matrix Profile VIII: Domain Agnostic Online Semantic Segmentation at Superhuman Performance Levels. 
        Shaghayegh Gharghabi, Yifei Ding, Chin-Chia Michael Yeh, Kaveh Kamgar, Liudmila Ulanova, and Eamonn Keogh.
        ICDM 2017.
        DOI: `10.1109/ICDM.2017.21 <https://doi.org/10.1109/ICDM.2017.21>`_
        `Alternative Reference <http://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`_
    """
    return _pygauss.cac(profile, index, window)

def segment(profile: ArrayLike, index: ArrayLike, window: int, num_reg: int = -1, ez:int = 5) -> List[int]:
    """
    Unsupervised semantic segmentation derived from a matrix profile.

    Parameters
    ----------
    profile: ArrayLike
        The array of distances obtained from a matrix profile analysis.
    
    index: ArrayLike
        The array of indices obtained from a matrix profile analysis.

    window: int
        The size of the window applied during the matrix profile analysis.
    
    num_reg: int, defaults to -1
        Maximum number of indices to obtain; when left unset or set to -1, it will return all 
        thoses indices found during the sementation analysis; when set to a positive number,
        it will return at most ``num_reg`` indices.
    
    ez: int, defaults to 5
        Defines the size of the exclusion zone, which is a multiplier over the window size.  

    Returns
    -------
    List of integers
        Sorted list of indices where the segments start.  The result is ordered always by significance, 
        that is, those index with strong changes in regime will be returned first.

    See Also
    --------
    cac
        Allows direct inspection of the implied ``Corrected Arc Crossings`` (CAC) curve.

    References
    ----------
    [1] Matrix Profile VIII: Domain Agnostic Online Semantic Segmentation at Superhuman Performance Levels. 
        Shaghayegh Gharghabi, Yifei Ding, Chin-Chia Michael Yeh, Kaveh Kamgar, Liudmila Ulanova, and Eamonn Keogh.
        ICDM 2017.
        DOI: `10.1109/ICDM.2017.21 <https://doi.org/10.1109/ICDM.2017.21>`_
        `Alternative Reference <http://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`_
    """
    return _pygauss.segment(profile, index, window, num_reg, ez)

def snippets(ts: ArrayLike, snippet_size: int, num_snippets: int, window_size: Optional[int] = None)-> List[Any]:
    import numpy as np 
    import matrixprofile as morg 
    return morg.discover.snippets(np.array(ts), snippet_size, num_snippets)
    
def snippets_int(ts: ArrayLike, snippet_size: int, num_snippets: int, window_size: Optional[int] = None)-> List[Snippet]:
    return _pygauss.snippets(ts, snippet_size, num_snippets, window_size)

__all__ = [
    "Snippet", "MatrixProfile", "MatrixProfileLR", 
    "mass", "matrix_profile", "matrix_profile_lr", 
    "mpdist_vect", "cac", "segment",
    "snippets", "snippets_int"
]

# Input: d # distance between subsequences X, Y
# Input: m # length of each subsequence
# Input: std_X # standard deviation of X
# Input: std_Y # standard deviation of Y
# Input: std_n # standard deviation of noise    
# return sqrt(dˆ2 - (2 + 2m) * std_nˆ2 / max(std_X, std_Y)ˆ2)