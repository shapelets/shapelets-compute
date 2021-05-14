from __future__ import annotations
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

DistanceType = Literal['additive_symm_chi', 'avg_l1_linf', 'bhattacharyya', 'canberra',
                       'chebyshev', 'clark', 'cosine', 'czekanowski', 'dice', 'divergence',
                       'dtw', 'euclidean', 'fidelity', 'gower', 'hamming', 'harmonic_mean', 'hellinger',
                       'innerproduct', ' intersection', 'jaccard', 'jeffrey', 'jensen_difference', 'jensen_shannon',
                       'k_divergence', 'kulczynski', 'kullback', 'kumar_johnson', 'kumar_hassebrook', 'lorentzian',
                       'manhattan', 'matusita', 'minkowski', 'mpdist',
                       'neyman', 'pearson', 'prob_symmetric_chi', 'sbd', 'soergel', 'sorensen', 'square_chord',
                       'squared_chi', 'squared_euclidean', 'taneja', 'topsoe', 'wave_hedges',
                       'ruzicka', 'motyka', 'tanimoto']

__dst_map = {
    'additive_symm_chi': _pygauss.DistanceType.Additive_Symm_Chi,
    'avg_l1_linf': _pygauss.DistanceType.Avg_L1_Linf,
    'bhattacharyya': _pygauss.DistanceType.Bhattacharyya,
    'canberra': _pygauss.DistanceType.Canberra,
    'chebyshev': _pygauss.DistanceType.Chebyshev,
    'clark': _pygauss.DistanceType.Clark,
    'cosine': _pygauss.DistanceType.Cosine,
    'czekanowski': _pygauss.DistanceType.Czekanowski,
    'dice': _pygauss.DistanceType.Dice,
    'divergence': _pygauss.DistanceType.Divergence,
    'dtw': _pygauss.DistanceType.DTW,
    'euclidean': _pygauss.DistanceType.Euclidean,
    'fidelity': _pygauss.DistanceType.Fidelity,
    'gower': _pygauss.DistanceType.Gower,
    'hamming': _pygauss.DistanceType.Hamming,
    'harmonic_mean': _pygauss.DistanceType.Harmonic_mean,
    'hellinger': _pygauss.DistanceType.Hellinger,
    'innerproduct': _pygauss.DistanceType.Innerproduct,
    'intersection': _pygauss.DistanceType.Intersection,
    'jaccard': _pygauss.DistanceType.Jaccard,
    'jeffrey': _pygauss.DistanceType.Jeffrey,
    'jensen_difference': _pygauss.DistanceType.Jensen_Difference,
    'jensen_shannon': _pygauss.DistanceType.Jensen_Shannon,
    'k_divergence': _pygauss.DistanceType.K_Divergence,
    'kulczynski': _pygauss.DistanceType.Kulczynski,
    'kullback': _pygauss.DistanceType.Kullback,
    'kumar_johnson': _pygauss.DistanceType.Kumar_Johnson,
    'kumar_hassebrook': _pygauss.DistanceType.Kumar_Hassebrook,
    'lorentzian': _pygauss.DistanceType.Lorentzian,
    'manhattan': _pygauss.DistanceType.Manhattan,
    'matusita': _pygauss.DistanceType.Matusita,
    'minkowski': _pygauss.DistanceType.Minkowski,
    'mpdist': _pygauss.DistanceType.MPDist,
    'neyman': _pygauss.DistanceType.Neyman,
    'pearson': _pygauss.DistanceType.Pearson,
    'prob_symmetric_chi': _pygauss.DistanceType.Prob_Symmetric_Chi,
    'sbd': _pygauss.DistanceType.SBD,
    'soergel': _pygauss.DistanceType.Soergel,
    'sorensen': _pygauss.DistanceType.Sorensen,
    'square_chord': _pygauss.DistanceType.Square_Chord,
    'squared_chi': _pygauss.DistanceType.Squared_Chi,
    'squared_euclidean': _pygauss.DistanceType.Squared_Euclidean,
    'taneja': _pygauss.DistanceType.Taneja,
    'topsoe': _pygauss.DistanceType.Topsoe,
    'wave_hedges': _pygauss.DistanceType.Wave_Hedges,
    'ruzicka': _pygauss.DistanceType.Ruzicka,
    'motyka': _pygauss.DistanceType.Motyka,
    'tanimoto': _pygauss.DistanceType.Tanimoto
}


def __convert_dst_type(type: DistanceType):
    if type in __dst_map:
        return __dst_map[type]
    raise ValueError("Unknown distance type")

# TODO: This construct is possible; one can send it through kwargs
#       and get it executed directly from the c++ side.
# class CustomDistanceFn(TypedDict):
# is_symetric: bool
# same_dimensionality: bool
# fn: Callable[[ShapeletsArray, ShapeletsArray], ShapeletsArray]

def pdist(tss: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray:
    """
    Pairwise distances between observations in n-dimensional space.

    This operation will operate in *all againsts all* mode whereby, if there 
    are M column vectors in tss, the output will be a square matrix of MxM 
    elements.  It is implicit by this set up that all the time series are of 
    the same length.

    Parameters
    ----------
    tss: 2-D array, nxM
        M column vectors of n elements.

    metric: DistanceType
        Selects the distance or simmilarity function to run.  
    
    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (MxM) where each element :math:`x_{ij}` represents the 
        results of applying `metric` to the i-th and j-th column.

    Notes
    -----
    Some metrics require parameters; for example, the :obj:`~shapelets.compute.distances.minkowski` 
    distance requires an arbitrary exponent, ``p``.  For those cases, use kwargs to pass 
    additional information required to run the metric.

    Examples
    --------
    Run :obj:`~shapelets.compute.distances.minkowski` distance over the 8 unitary vectors on 
    a three dimensional space:

    >>> import shapelets.compute as sc 
    >>> b = sc.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], dtype="float32") 
    >>> b.display()
    [3 8 1 1]
        0.0000     0.0000     0.0000     0.0000     1.0000     1.0000     1.0000     1.0000 
        0.0000     0.0000     1.0000     1.0000     0.0000     0.0000     1.0000     1.0000 
        0.0000     1.0000     0.0000     1.0000     0.0000     1.0000     0.0000     1.0000 

    >>> sc.distances.pdist(b, 'minkowski', p=0.5)
    [8 8 1 1]
        0.0000     1.0000     1.0000     4.0000     1.0000     4.0000     4.0000     9.0000 
        1.0000     0.0000     4.0000     1.0000     4.0000     1.0000     9.0000     4.0000 
        1.0000     4.0000     0.0000     1.0000     4.0000     9.0000     1.0000     4.0000 
        4.0000     1.0000     1.0000     0.0000     9.0000     4.0000     4.0000     1.0000 
        1.0000     4.0000     4.0000     9.0000     0.0000     1.0000     1.0000     4.0000 
        4.0000     1.0000     9.0000     4.0000     1.0000     0.0000     4.0000     1.0000 
        4.0000     9.0000     1.0000     4.0000     1.0000     4.0000     0.0000     1.0000 
        9.0000     4.0000     4.0000     1.0000     4.0000     1.0000     1.0000     0.0000     
    """
    return _pygauss.pdist(tss, __convert_dst_type(metric), **kwargs)


def cdist(xa: ArrayLike, xb: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray:
    """
    Compute distance between each pair of the two collections of inputs.

    This function will compute the distance of simmilarity metric for every column vector 
    in xa against all column vectors in xb.  

    if xa and xb column vectors are not of the same length, and the algorithm requires 
    same length vectors, zero padding will be used to adjust the sizes. 

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    metric: DistanceType
        Selects the distance or simmilarity function to run. 

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        results of applying `metric` to the i-th column of xa against the j-th 
        column of xb.

    Notes
    -----
    Some metrics require parameters; for example, the :obj:`~shapelets.compute.distances.minkowski` 
    distance requires an arbitrary exponent, ``p``.  For those cases, use kwargs to pass 
    additional information required to run the metric.    

    Examples
    --------
    Run :obj:`~shapelets.compute.distances.minkowski` distance between the 3 norm vectors and the 8 
    unitary vectors on a three dimensional space:

    >>> import shapelets.compute as sc
    >>> a = sc.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")
    >>> a.display()
    [3 3 1 1]
        0.0000     0.0000     1.0000 
        0.0000     1.0000     0.0000 
        1.0000     0.0000     0.0000   

    >>> b = sc.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], dtype="float32")
    >>> b.display()
    [3 8 1 1]
        0.0000     0.0000     0.0000     0.0000     1.0000     1.0000     1.0000     1.0000 
        0.0000     0.0000     1.0000     1.0000     0.0000     0.0000     1.0000     1.0000 
        0.0000     1.0000     0.0000     1.0000     0.0000     1.0000     0.0000     1.0000     

    >>> sc.distances.cdist(a, b, 'minkowski', p=0.5)
    [3 8 1 1]
        1.0000     0.0000     4.0000     1.0000     4.0000     1.0000     9.0000     4.0000 
        1.0000     4.0000     0.0000     1.0000     4.0000     9.0000     1.0000     4.0000 
        1.0000     4.0000     4.0000     9.0000     0.0000     1.0000     1.0000     4.0000 

    """
    return _pygauss.cdist(xa, xb, __convert_dst_type(metric), **kwargs)

def euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute euclidian distance between each pair of the two collections of inputs.

    The euclidian distance belongs to the the :math:`L_p` Minkowski family, with exponent 2.0 and 
    it is defined as:
        
        .. math::

            d_{Euc} = \sqrt{\sum_{i=1}^{d}(P_i-Q_i)^2}

    This function will compute the euclidian distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        euclidian distance from the i-th column of xa to the j-th column of xb.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Euclidean)

def manhattan(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute manhattan distance between each pair of the two collections of inputs.

    The manhattan, also known as the *City Block*, *rectiliniar* or *taxi cab* distance, 
    belongs to the the :math:`L_p` Minkowski family, with exponent 1.0 and it is defined as:
        
        .. math::

            d_{man} = \sum_{i=1}^{d} \left | P_i-Q_i \right |

    This function will compute the manhattan distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        manhattan distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Taxicab Geometry An Adventure in Non-Euclidean Geometry 
        Krause E.F.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Manhattan)    

def minkowski(a: ArrayLike, b: ArrayLike, p: float) -> ShapeletsArray:
    r"""
    Compute minkowski distance between each pair of the two collections of inputs.

    The minkowski distance is defined as:
        
        .. math::

            d_{p} =\sqrt[p]{ \sum_{i=1}^{d}\left | P_i-Q_i \right |^p }

    This function will compute the minkowski distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    p: float
        Exponent

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        minkowski distance from the i-th column of xa to the j-th column of xb.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Minkowski, p=p)

def chebyshev(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute chebyshev distance between each pair of the two collections of inputs.

    The chebyshev, named after *Pafnuty Lvovich Chebyshev* and also knwon as *the chessboard distance*, 
    belongs to the the :math:`L_p` Minkowski family, with exponent ``p`` goes to infinity.  It is defined as:
        
        .. math::

            d_{chev} = \underset{i}{max} \left | P_i-Q_i \right |

    This function will compute the chebyshev for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        chebyshev distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Classification, Parameter Estimation and State Estimation: An Engineering Approach using MATLAB. 
        David M. J. Tax, Robert Duin, and Dick De Ridder (2004)
        John Wiley and Sons. 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Chebyshev)

def sorensen(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Sørensen distance between each pair of the two collections of inputs.

    The sorensen, also known as *Bray-Curtis* distance, widely used in ecology,
    belongs to the the :math:`L_1` family.  It is defined as:
        
        .. math::

            d_{sor} = \frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}(P_i+Q_i)}

    This function will compute the sorensen for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        sorensen distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] A method of establishing groups of equal amplitude in plant sociology based on similarity of species and its application to analyses of the vegetation on Danish commons. 
        Sørensen, T. (1948) 
        Biologiske Skrifter / Kongelige Danske Videnskabernes Selskab, 5 (4): 1-34. 
    [2] Dictionary of Distances
        Deza E. and Deza M.M., Elsevier, 2006 
    [3] Introduction to Similarity Searching in Chemistry
        Monev V., MATCH Commun. Math. Comput. Chem. 51 pp. 7-38 , 2004
    [4] An ordination of the upland forest of the southern Winsconsin. 
        Bray J. R., Curtis J. T., 1957. 
        Ecological Monographies, 27, 325-349.         
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Sorensen)

def gower(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Gower distance between each pair of the two collections of inputs.

    The gower distance belongs to the the :math:`L_1` family.  It is defined as:
        
        .. math::

            d_{gow} =\frac{1}{d} \sum_{i=1}^{d}\left | P_i-Q_i \right |

    This function will compute the gower distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        gower distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] General Coefficient of Similarity and Some of Its Properties
        Gower, J.C., Biometrics 27, pp857-874 1971 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Gower)

def soergel(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Soergel distance between each pair of the two collections of inputs.

    The soergel distance belongs to the the :math:`L_1` family.  It is defined as:
        
        .. math::

            d_{sg} =\frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}max(P_i, Q_i)} 

    This function will compute the soergel distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        soergel distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Introduction to Similarity Searching in Chemistry, 
        Monev V.
        MATCH Commun. Math. Comput. Chem. 51 pp. 7-38 , 2004
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Soergel)

def kulczynski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Kulczynski distance between each pair of the two collections of inputs.

    The Kulczynski distance belongs to the the :math:`L_1` family.  It is defined as:

    .. math::

        d_{kul} =\frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}min(P_i, Q_i)} 

    This function will compute the Kulczynski distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Kulczynski distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kulczynski)

def canberra(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Canberra distance between each pair of the two collections of inputs.

    The canberra distance belongs to the the :math:`L_1` family.  It is defined as:

    .. math::

        d_{can} =\sum_{i=1}^{d}\frac{\left | P_i-Q_i \right |}{P_i+Q_i}

    It resembles :obj:`~shapelets.compute.distances.sorensen` but normalizes the absolute 
    difference of the individual level. It is known to be very sensitive to small changes 
    near zero.

    This function will compute the canberra distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        canberra distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 

    [2] Classification. 
        Gordon, A.D.,
        2nd edition London-New York 1999 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Canberra)

def lorentzian(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Lorentzian distance between each pair of the two collections of inputs.

    The lorentzian distance belongs to the the :math:`L_1` family.  It is defined as:

    .. math::

        d_{lor} =\sum_{i=1}^{d}ln(1+\left | P_i - Q_i \right |)

    It applies the natural logarithm to the absolute difference, adding one to guarantee 
    non negativity and to eschew the log of zero.

    This function will compute the lorentzian distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        lorentzian distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Lorentzian)

def intersection(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Intersection simmilarity between each pair of the two collections of inputs.

    The intersection simmilarity is defined as:

    .. math::

       s_{is} =\sum_{i=1}^{d}\min(P_i,Q_i)

    This function will compute the intersection simmilarity for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        intersection simmilarity from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Pattern Classification, 2nd ed. 
        Duda, R.O., Hart, P.E., and Stork, D.G., P
        Wiley, 2001
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Intersection)

def wavehedges(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Wave Hedges distance between each pair of the two collections of inputs.

    The wavehedges distance is defined as:

    .. math::

       d_{wh} =\sum_{i=1}^{d}\frac{\left | P_i - Q_i \right |}{max(P_i, Q_i)}

    This function will compute the wavehedges distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        wavehedges distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] An empirical modication to linear wave theory
        Hedges, T.S., 1976
        Proc. Inst. Civ. Eng. , 61, 575-579. 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Wave_Hedges)

def czekanowski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Czekanowski simmilarity between each pair of the two collections of inputs.

    The czekanowski simmilarity is defined as:

    .. math::

       s_{cze} = \frac{2 \sum_{i-1}^{d}\min(P_i, Q_i}{\sum_{i-1}^{d}(P_i+Q_i)}  

    This function will compute the czekanowski simmilarity for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        czekanowski simmilarity from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Pattern Classification, 2nd ed. 
        Duda, R.O., Hart, P.E., and Stork, D.G., P
        Wiley, 2001
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Czekanowski)

def ruzicka(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Ruzicka simmilarity between each pair of the two collections of inputs.

    The ruzicka simmilarity is defined as:

    .. math::

       s_{ruz} = \frac{\sum_{i-1}^{d}\min(P_i, Q_i)}{\sum_{i-1}^{d}\max(P_i, Q_i)}

    This function will compute the ruzicka simmilarity for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        ruzicka simmilarity from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Ruzicka)

def motyka(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Motyka simmilarity between each pair of the two collections of inputs.

    The motyka simmilarity is defined as:

    .. math::

       s_{mot} = \frac{\sum_{i-1}^{d}\min(P_i, Q_i)}{\sum_{i-1}^{d}(P_i + Q_i)}

    This function will compute the motyka simmilarity for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        motyka simmilarity from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Motyka)

def tanimoto(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute Tanimoto distance between each pair of the two collections of inputs.

    The tanimoto distance is defined as:

    .. math::

       d_{tani} = \frac{\sum_{i-1}^{d}P_i + \sum_{i-1}^{d}Q_i - 2 \sum_{i-1}^{d}\min(P_i,Q_i)}{\sum_{i-1}^{d}P_i + \sum_{i-1}^{d}Q_i - \sum_{i-1}^{d}\min(P_i,Q_i)}

    This function will compute the tanimoto distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        tanimoto distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Pattern Classification, 2nd ed. 
        Duda, R.O., Hart, P.E., and Stork, D.G.
        Wiley, 2001 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Tanimoto)

def innerproduct(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the inner product as a simmilarity measure between each pair of the two collections of inputs.

    .. math::

       s_{ip} =\sum_{i=1}^{d}P_iQ_i

    This function will compute the inner product for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        innerproduct from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Pattern Classification, 2nd ed. 
        Duda, R.O., Hart, P.E., and Stork, D.G.
        Wiley, 2001
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Innerproduct)

def harmonic_mean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the harmonic mean as a simmilarity measure between each pair of the two collections of inputs.

    .. math::

       s_{ip} =2 \sum_{i=1}^{d}\frac{P_iQ_i}{P_i+Q_i}

    This function will compute the harmonic mean for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        harmonic mean from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Harmonic_mean)

def cosine(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the cosine simmilarity measure between each pair of the two collections of inputs.

    .. math::

       s_{cos} = \frac{\sum_{i=1}^{d}P_iQ_i}{\sqrt{\sum_{i=1}^{d}P_i^2}\sqrt{\sum_{i=1}^{d}Q_i^2}}

    This metric is also known as the cosine coefficient because it measures the angle between 
    two vectors and thus often called the angular metric.  Other names for the cosine coefficient 
    include Ochiai and Carbo.

    This function will compute the cosine simmilarity for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        cosine simmilarity from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006
    [2] Introduction to Similarity Searching in Chemistry, 
        Monev V. 
        MATCH Commun. Math. Comput. Chem. 51 pp. 7-38 , 2004 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Cosine)

def kumarhassebrook(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the peak-to-correlation energy (PCE) as a simmilarity measure between 
    each pair of the two collections of inputs.

    .. math::

       s_{pce} = \frac{\sum_{i=1}^{d}P_iQ_i}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 - \sum_{i=1}^{d}P_iQ_i }

    This function will compute the PCE for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        PCE from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Performance measures for correlation filters 
        B. V. K. Vijaya Kumar and L. G. Hassebrook
        Appl. Opt. 29, 2997-3006 (1990).
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kumar_Hassebrook)

def jaccard(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Jaccard distance between each pair of the two collections of inputs.

    .. math::

       d_{jac} = \frac{\sum_{i=1}^{d}(P_i-Q_i)^2}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 - \sum_{i=1}^{d}P_iQ_i }

    This function will compute the jaccard distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        jaccard distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Étude comparative de la distribution florale dans une portion des Alpes et des Jura. 
        Jaccard P.
        Bulletin del la Société Vaudoise des Sciences Naturelles 37, 1901, 547-579.
    [2] IBM Internal Report 17th Nov. 1957   
        Tanimoto, T.T. (1957) 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jaccard)

def dice(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Dice distance between each pair of the two collections of inputs.

    .. math::

       d_{dice} = \frac{\sum_{i=1}^{d}(P_i-Q_i)^2}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 }

    This function will compute the dice distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        dice distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Measures of the amount of ecologic association between species
        Dice, L. R.
        Ecology, 26:297-302, 1945 

    [2] Measuring of interspecific association and similarity between communities. 
        Morisita M. 
        Mem. Fac. Sci. Kyushu Univ. Ser. E (Biol.) 3:65-80, 1959.

    [3] Introduction to Similarity Searching in Chemistry, 
        Monev V., 
        MATCH Commun. Math. Comput. Chem. 51 pp. 7-38 , 2004
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Dice)

def fidelity(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute sum of geometric means as a simmilarity measure between 
    each pair of the two collections of inputs.

    The sum of geometric means is referred to as Fidelity similarity, 
    a.k.a. Bhattacharyya coefficient or Hellinger affinity:

    .. math::

       s_{fid} = \sum_{i=1}^{d}\sqrt{P_iQ_i}

    This function will compute the sum of geometric means for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        sum of geometric means from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Fidelity)

def bhattacharyya(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Bhattacharyya distance between each pair of the two collections of inputs.

    This distance, which is valued between 0 and 1, provides also bounds on the Bayes 
    misclassification probability:

    .. math::

       d_{b} = -\ln \sum_{i=1}^{d}\sqrt{P_iQ_i}

    This function will compute the bhattacharyya distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        bhattacharyya distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] On a measure of divergence between two statistical populations defined by probability distributions
        A. Bhattacharyya, 
        Bull. Calcutta Math. Soc., vol. 35, pp. 99–109, 1943
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Bhattacharyya)

def hellinger(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Hellinger distance between each pair of the two collections of inputs.

    .. math::

       d_{H} = 2 \sqrt{1 - \sum_{i=1}^{d}\sqrt{P_iQ_i}}

    This function will compute the hellinger distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        hellinger distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Hellinger)

def matusita(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Matusita distance between each pair of the two collections of inputs.

    .. math::

       d_{M} = \sqrt{2 - 2\sum_{i=1}^{d}\sqrt{P_iQ_i}}

    This function will compute the matusita distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        matusita distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Decision rules, based on the distance, for problems of fit, two samples, and estimation
        K. Matusita
        Ann. Math. Statist. 26 (1955) 631–640
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Matusita)

def square_chord(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Square Chord distance between each pair of the two collections of inputs.

    .. math::

       d_{sqc} = \sum_{i=1}^{d}(\sqrt{P_i} - \sqrt{Q_i})^2

    This function will compute the square chord distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        square chord distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] A statistical approach to evaluating distance metrics and analog assignments for pollen records
        Gavin D.G., Oswald W.W., Wahl, E.R., and Williams J.W., 
        Quaternary Research 60, pp 356–367, 2003
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Square_Chord)

def squared_euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the squared Euclidian distance between each pair of the two collections of inputs.

    .. math::

       d_{sqe} = \sum_{i=1}^{d}(P_i-Q_i)^2

    This function will compute the squared Euclidian distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        squared Euclidian chord distance from the i-th column of xa to the j-th column of xb.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Squared_Euclidean)

def pearson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Pearson :math:`\chi^2` distance between each pair of the two collections of inputs.

    .. math::

       d_{pearson} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{Q_i}

    *Please note this metric is not simmetric!*

    This function will compute the Pearson :math:`\chi^2` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Pearson :math:`\chi^2` distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] On the Criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonable supposed to have arisen from random sampling
        Pearson, K. 
        Phil. Mag.,1900, 50, 157-172.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Pearson)

def neyman(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Neyman :math:`\chi^2` distance between each pair of the two collections of inputs.

    .. math::

       d_{neyman} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i}

    *Please note this metric is not simmetric!*

    This function will compute the Neyman :math:`\chi^2` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Neyman :math:`\chi^2` distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Contributions to the theory of the :math:`\chi^2` test. 
        J. Neyman. 
        In Proceedings of the First Berkley Symposium on Mathematical Statistics and Probability, 1949.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Neyman)

def squared_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Squared :math:`\chi^2`, or triangular discrimination, distance between each 
    pair of the two collections of inputs.

    .. math::

       d_{SqChi} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i+Q_i}

    This function will compute the Squared :math:`\chi^2` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Squared :math:`\chi^2` distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] New Inequalities for Jeffreys Divergence Measure
        S. S. DRAGOMIR, J. SUNDE and C. BUSE, 
        Tamsui Oxford Journal of Mathematical Sciences, 16(2)(2000), 295-309.

    [2] Some Inequalities for Information Divergence and Related Measures of Discrimination
        F. TOPSØE
        IEEE Trans. on Inform. Theory, IT46(2000), 1602-1609

    [3] A statistical approach to evaluating distance metrics and analog assignments for pollen records
        Gavin D.G., Oswald W.W., Wahl, E.R., and Williams J.W., 
        Quaternary Research 60, pp 356–367, 2003
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Squared_Chi)

def prob_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Probabilistic Symmetric :math:`\chi^2` distance between each pair of the two collections of inputs.

    This distance is equivalent to Sangvi :math:`\chi^2` distance between populations: 

    .. math::

       d_{PChi} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i+Q_i}

    This function will compute the Probabilistic Symmetric :math:`\chi^2` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Probabilistic Symmetric :math:`\chi^2` distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Prob_Symmetric_Chi)

def divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the divergence distance (non-metric) between each pair of the two collections of inputs.

    .. math::

       d_{divergence} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{(P_i+Q_i)^2}

    This function will compute the divergence distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        divergence distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Multidimensional Scaling
        Cox, T.F. and Cox, M.A.A.
        Chapman & Hall/CRC 2nd ed. 2001
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Divergence)

def clark(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Clark distance between each pair of the two collections of inputs.

    .. math::

       d_{clark} = \sqrt{\sum_{i=1}^{d} \left ( \frac{ \left |P_i-Q_i \right |}{P_i+Q_i} \right )^2}

    This function will compute the clark distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        clark distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Clark)

def additive_symm_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Additive Symmetric :math:`\chi^2` distance between each pair of the two collections of inputs.

    .. math::

       d_{adchi} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2(P_i+Q_i)}{P_iQ_i} 

    Please note that :math:`d_{adchi}(P,Q) = d_p(P,Q) + d_p(Q,P)`

    This function will compute the additive symmetric :math:`\chi^2` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        additive symmetric :math:`\chi^2` distance from the i-th column of xa 
        to the j-th column of xb.

    References
    ----------
    [1] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 

    [2] Similarity Search The Metric Space Approach
        Zezula P., Amato G., Dohnal V., and Batko M.
        Springer, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Additive_Symm_Chi)

def kullback(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Kullback-Leibler (KL) distance between each pair of the two collections of inputs.

    This distance is also known as KL divergence, relative entropy, or information deviation:

    .. math::

       d_{kl} = \sum_{i=1}^{d}P_i \, \textup{ln}\frac{P_i}{Q_i}

    *Please note this distance is not simmetric!*

    This function will compute the KL distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        KL distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] On information and sufficiency
        S. Kullback, R.A. Leibler, 
        Ann. Math. Statist. 22 (1951) 79–86.

    [2] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006 
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kullback)

def jeffrey(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Jeffrey distance between each pair of the two collections of inputs.

    Jeffrey distance is the symmetric form of the KL divergence using the addition method:

    .. math::

       d_{j} = \sum_{i=1}^{d}(P_i-Q_i)\textup{ln}\frac{P_i}{Q_i}

    This function will compute the jeffrey distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    See Also
    --------
    kullback

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        jeffrey distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] On information and sufficiency
        S. Kullback, R.A. Leibler
        Ann. Math. Statist. 22 (1951) 79–86. 

    [2] An Invariant Form for the Prior Probability in Estimation Problems
        JEFFREYS, H. (1946)
        Proc. Roy. Soc. Lon., Ser. A, 186, 453-461. 

    [3] Generalized Information Measures and Their Applications
        TANEJA. I.J. (2001)
        `Online book <www.mtm.ufsc.br/~taneja/book/book.html>`_
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jeffrey)

def k_divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the K-Divergence distance between each pair of the two collections of inputs.

    .. math::

       d_{Kdiv} = \sum_{i=1}^{d}P_i \, \textup{ln}\frac{2P_i}{P_i+Q_i}

    *Please note this distance is not simmetric!*

    This function will compute the K-Divergence distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    See Also
    --------
    topsoe
    jensen_shannon

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        K-Divergence distance from the i-th column of xa to the j-th column of xb.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.K_Divergence)

def topsoe(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Topsoe distance between each pair of the two collections of inputs.

    .. math::

       d_{top} = \sum_{i=1}^{d} \left [ P_i\,\textup{ln}\left(\frac{2P_i}{P_i+Q_i}\right)+Q_i\,\textup{ln}\left(\frac{2Q_i}{P_i+Q_i}\right) \right ]

    Topsoe distance, also called *information statistics*, is the symmetric 
    form of :obj:`~shapelets.compute.distances.k_divergence`

    This function will compute the topsoe distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    See Also
    --------
    k_divergence
    jensen_shannon

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        topsoe distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] A statistical approach to evaluating distance metrics and analog assignments for pollen records
        Gavin D.G., Oswald W.W., Wahl, E.R., and Williams J.W.
        Quaternary Research 60, pp 356–367, 2003 
    [2] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006  
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Topsoe)

def jensen_shannon(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Jensen-Shannon divergence distance between each pair of the two collections of inputs.

    .. math::

       d_{JS} = \frac{1}{2}\left[\sum_{i=1}^{d} P_i\,\textup{ln}\left(\frac{2P_i}{P_i+Q_i}\right)+\sum_{i=1}^{d} Q_i\,\textup{ln}\left(\frac{2Q_i}{P_i+Q_i}\right)\right]  

    This distance uses the average method to make :obj:`~shapelets.compute.distances.k_divergence` symmetric.

    This function will compute the Jensen-Shannon divergence distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    See Also
    --------
    k_divergence
    topsoe

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Jensen-Shannon divergence distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Generalized Information Measures and Their Applications
        TANEJA. I.J. (2001)
        `Online book <www.mtm.ufsc.br/~taneja/book/book.html>`_
    [2] Dictionary of Distances
        Deza E. and Deza M.M.
        Elsevier, 2006  
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jensen_Shannon)

def jensen_difference(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Jensen difference distance between each pair of the two collections of inputs.

    .. math::

       d_{jd} = \sum_{i=1}^{d}\left [\frac{P_i\textup{ln} P_i + Q_i\textup{ln}Q_i}{2} -\left(\frac{P_i+Q_i}{2} \right )\textup{ln}\left(\frac{P_i+Q_i}{2} \right ) \right ]

    Sibson studied the idea of information radius for a measure arising due to concavity property of Shannon's entropy and 
    introduced the Jensen difference. 
    
    This function will compute the Jensen difference distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Jensen difference distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Information Radius
        SIBSON, R. (1969)
        Z. Wahrs. und verw Geb., 14, 149-160

    [2] Generalized Information Measures and Their Applications
        TANEJA. I.J. (2001)
        `Online book <www.mtm.ufsc.br/~taneja/book/book.html>`_        
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jensen_Difference)

def taneja(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Taneja distance between each pair of the two collections of inputs.

    .. math::

       d_{TJ} = \sum_{i=1}^{d}\left(\frac{P_i+Q_i}{2}\right )\textup{ln}\left(\frac{P_i+Q_i}{2\sqrt{P_iQ_i}} \right )

    This function will compute the Taneja distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Taneja distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] New Developments in Generalized Information Measures
        TANEJA, I.J. (1995), 
        Chapter in: Advances in Imaging and Electron Physics, Ed. P.W. Hawkes, 91, 37-135.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Taneja)

def kumar_johnson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Kumar-Johnson distance between each pair of the two collections of inputs.

    .. math::

       d_{kj} = \sum_{i=1}^{d}\left ( \frac{(P_i^2-Q_i^2)^2}{2(P_iQ_i)^{3/2}} \right )

    This function will compute the Kumar-Johnson distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Kumar-Johnson distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] On a symmetric divergence measure and information inequalities
        Kumar P. and Johnson A., 2005, 
        Journal of Inequalities in pure and applied Mathematics, Vol 6, Issue 3, article 65
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kumar_Johnson)

def avg_l1_linf(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Compute the Average :math:`L_1 - L_\infty` distance between each pair of the two collections of inputs.

    .. math::

       d_{ACC} = \frac{\sum_{i=1}^{d}\left |P_i-Q_i\right |+\underset{i}{max}\left|P_i-Q_i\right|}{2}

    This function will compute the Average :math:`L_1 - L_\infty` distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        Average :math:`L_1 - L_\infty` distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] Taxicab Geometry An Adventure in Non-Euclidean Geometry
        Krause E.F.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Avg_L1_Linf)

def hamming(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist no formula
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Hamming)

def mpdist(a: ArrayLike, b: ArrayLike, w: int, threshold: float = 0.05) -> ShapeletsArray:
    r"""
    Calculates the distance between time series using Matrix Profile, MPDist.

    It considers two time series to be similar if they share many similar subsequences, 
    regardless of the order of matching subsequences. MPdist is robust to spikes, warping, 
    linear trends, dropouts, wandering baseline and missing values, issues that are 
    common outside of benchmark datasets.

    This function will compute the MPDist distance for every column vector 
    in xa against all column vectors in xb.  If the length of the column vectors is 
    not the same, the smaller vectors will be padded with zeros.

    Parameters
    ----------
    xa: 2-D matrix, nxA
        A column vectors of length n.
    
    xb: 2-D matrix, mxB
        B column vectors of length m.

    Returns
    -------
    ShapeletsArray
        A new 2-D matrix (AxB) where each element :math:`x_{ij}` represents the 
        MPDist distance from the i-th column of xa to the j-th column of xb.

    References
    ----------
    [1] `Matrix Profile XII: MPdist`_ A Novel Time Series Distance Measure to Allow Data Mining in More Challenging Scenarios.
    *Gharghabi S, Imani S, Bagnall A, Darvishzadeh A, Keogh E.* 2018 IEEE International Conference on Data Mining (ICDM). 2018.

    .. _`Matrix Profile XII: MPdist`: https://sites.google.com/site/mpdistinfo/
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.MPDist, w=w, threshold=threshold)

def dtw(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Calculates the Dynamic Time Warping Distance.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.DTW)

def sbd(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Calculates the Shape-Based distance (SBD).
    
    It computes the normalized cross-correlation and it returns 1.0 minus the value 
    that maximizes the correlation value between each pair of time series.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.SBD)