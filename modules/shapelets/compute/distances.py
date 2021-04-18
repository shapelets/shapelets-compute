"""
Distances Documentation
-----------------------

blab bla balba 

"""
from __future__ import annotations
from typing import Literal
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

DistanceType = Literal['additive_symm_chi', 'avg_l1_linf', 'bhattacharyya', 'canberra',
                       'chebyshev', 'clark', 'cosine', 'czekanowski', 'dice', 'divergence',
                       'dtw', 'euclidean', 'fidelity', 'gower', 'hamming', 'harmonic_mean', 'hellinger',
                       'innerproduct', ' intersection', 'jaccard', 'jeffrey', 'jensen_difference', 'jensen_shannon',
                       'k_divergence', 'kulczynski', 'kullback', 'kumar_johnson', 'kumar_hassebrook', 'lorentzian',
                       'manhattan', 'matusita', 'max_symmetric_chi', 'min_symmetric_chi', 'minkowski', 'mpdist',
                       'neyman', 'pearson', 'prob_symmetric_chi', 'sbd', 'soergel', 'sorensen', 'square_chord',
                       'squared_chi', 'squared_euclidean', 'taneja', 'topsoe', 'vicis_wave_hedges', 'wave_hedges']

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
    'max_symmetric_chi': _pygauss.DistanceType.Max_Symmetric_Chi,
    'min_symmetric_chi': _pygauss.DistanceType.Min_Symmetric_Chi,
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
    'vicis_wave_hedges': _pygauss.DistanceType.Vicis_Wave_Hedges,
    'wave_hedges': _pygauss.DistanceType.Wave_Hedges
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
    """
    return _pygauss.pdist(tss, __convert_dst_type(metric), **kwargs)


def cdist(xa: ArrayLike, xb: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray:
    """
    Compute distance between each pair of the two collections of inputs.
    """
    return _pygauss.cdist(xa, xb, __convert_dst_type(metric), **kwargs)


def additive_symm_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{adchi} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2(P_i+Q_i)}{P_iQ_i}  

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Additive_Symm_Chi)


def avg_l1_linf(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{ACC} = \frac{\sum_{i=1}^{d}\left |P_i-Q_i\right |+\underset{i}{max}\left|P_i-Q_i\right|}{2}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Avg_L1_Linf)


def bhattacharyya(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance
    
    .. math::

        d_{b} = -ln \sum_{i=1}^{d}\sqrt{P_iQ_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Bhattacharyya)


def canberra(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{can} =\sum_{i=1}^{d}\frac{\left | P_i-Q_i \right |}{P_i+Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Canberra)


def chebyshev(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{chev} = \underset{i}{max} \left | P_i-Q_i \right |

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Chebyshev)


def clark(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{clark} = \sqrt{\sum_{i=1}^{d} \left ( \frac{ \left |P_i-Q_i \right |}{P_i+Q_i} \right )^2}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Clark)


def cosine(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    simil.

    .. math::

        s_{cos} = \frac{\sum_{i=1}^{d}P_iQ_i}{\sqrt{\sum_{i=1}^{d}P_i^2}\sqrt{\sum_{i=1}^{d}Q_i^2}}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Cosine)


def czekanowski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{cze} =\frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}(P_i+Q_i)}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Czekanowski)


def dice(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance

    .. math::

        d_{dice} = \frac{\sum_{i=1}^{d}(P_i-Q_i)^2}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 }

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Dice)


def divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{divergence} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{(P_i+Q_i)^2}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Divergence)

def euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{Euc} = \sqrt{\sum_{i=1}^{d}(P_i-Q_i)^2}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Euclidean)


def fidelity(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    sim

    .. math::

        s_{fid} = \sum_{i=1}^{d}\sqrt{P_iQ_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Fidelity)


def gower(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{gow} =\frac{1}{d} \sum_{i=1}^{d}\left | P_i-Q_i \right |

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Gower)


def hamming(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist no formula
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Hamming)


def harmonic_mean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    simil

    .. math::

        s_{ip} =2 \sum_{i=1}^{d}\frac{P_iQ_i}{P_i+Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Harmonic_mean)


def hellinger(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance

    .. math::

        d_{H} = 2 \sqrt{1 - \sum_{i=1}^{d}\sqrt{P_iQ_i}}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Hellinger)


def innerproduct(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    simil.

    .. math::

        s_{ip} =\sum_{i=1}^{d}P_iQ_i

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Innerproduct)


def intersection(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{non\_is} =\frac{1}{2}\sum_{i=1}^{d}\left | P_i - Q_i \right |

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Intersection)


def jaccard(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance  

    .. math::

        d_{jac} = \frac{\sum_{i=1}^{d}(P_i-Q_i)^2}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 - \sum_{i=1}^{d}P_iQ_i }

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jaccard)


def jeffrey(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance

    .. math::

        d_{j} = \sum_{i=1}^{d}(P_i-Q_i)ln\frac{P_i}{Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jeffrey)


def jensen_difference(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{jd} = \sum_{i=1}^{d}\left [\frac{P_i\textup{ln} P_i + Q_i\textup{ln}Q_i}{2} -\left(\frac{P_i+Q_i}{2} \right )\textup{ln}\left(\frac{P_i+Q_i}{2} \right ) \right ]

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jensen_Difference)


def jensen_shannon(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{JS} = \frac{1}{2}\left[\sum_{i=1}^{d} P_i\,ln\left(\frac{2P_i}{P_i+Q_i}\right)+\sum_{i=1}^{d} Q_i\,ln\left(\frac{2Q_i}{P_i+Q_i}\right)\right]  

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Jensen_Shannon)


def k_divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{Kdiv} = \sum_{i=1}^{d}P_i \, ln\frac{2P_i}{P_i+Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.K_Divergence)

def kulczynski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""

    .. math::

        d_{kul} =\frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}min(P_i, Q_i)} 

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kulczynski)


def kullback(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{kl} = \sum_{i=1}^{d}P_i \, ln\frac{P_i}{Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kullback)


def kumar_johnson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{kj} = \sum_{i=1}^{d}\left ( \frac{(P_i^2-Q_i^2)^2}{2(P_iQ_i)^{3/2}} \right )

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kumar_Johnson)


def kumarhassebrook(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    simil.

    .. math::

        s_{pce} = \frac{\sum_{i=1}^{d}P_iQ_i}{\sum_{i=1}^{d}P_i^2 + \sum_{i=1}^{d}Q_i^2 - \sum_{i=1}^{d}P_iQ_i }

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Kumar_Hassebrook)


def lorentzian(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math:: 

        d_{lor} =\sum_{i=1}^{d}ln(1+\left | P_i - Q_i \right |)

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Lorentzian)


def manhattan(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{man} = \sum_{i=1}^{d} \left | P_i-Q_i \right |

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Manhattan)


def matusita(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{M} = \sqrt{2 - 2\sum_{i=1}^{d}\sqrt{P_iQ_i}}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Matusita)


def max_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{e5} = \textup{max}\left( \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i}\,,\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{Q_i} \right)

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Max_Symmetric_Chi)


def min_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{e5} = \textup{min}\left( \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i}\,,\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{Q_i} \right)

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Min_Symmetric_Chi)


def minkowski(a: ArrayLike, b: ArrayLike, p: float) -> ShapeletsArray:
    r"""
    .. math::

        d_{p} =\sqrt[p]{ \sum_{i=1}^{d}\left | P_i-Q_i \right |^p }

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Minkowski, p=p)


def neyman(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{neyman} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Neyman)


def pearson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{pearson} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Pearson)


def prob_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{PChi} = 2\sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i+Q_i}
        
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Prob_Symmetric_Chi)


def soergel(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{sg} =\frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}max(P_i, Q_i)} 

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Soergel)


def sorensen(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{sor} = \frac{\sum_{i=1}^{d}\left | P_i-Q_i \right |}{\sum_{i=1}^{d}(P_i+Q_i)}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Sorensen)


def square_chord(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dis

    .. math::

        d_{sqc} = \sum_{i=1}^{d}(\sqrt{P_i} - \sqrt{Q_i})^2

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Square_Chord)


def squared_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{SqChi} = \sum_{i=1}^{d}\frac{(P_i-Q_i)^2}{P_i+Q_i}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Squared_Chi)


def squared_euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{sqe} = \sum_{i=1}^{d}(P_i-Q_i)^2

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Squared_Euclidean)


def taneja(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{TJ} = \sum_{i=1}^{d}\left(\frac{P_i+Q_i}{2}\right )\textup{ln}\left(\frac{P_i+Q_i}{2\sqrt{P_iQ_i}} \right )

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Taneja)


def topsoe(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    distance

    .. math::

        d_{top} = \sum_{i=1}^{d} \left [ P_i\,ln\left(\frac{2P_i}{P_i+Q_i}\right)+Q_i\,ln\left(\frac{2Q_i}{P_i+Q_i}\right) \right ]

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Topsoe)


def vicis_wave_hedges(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    dist

    .. math::

        d_{emanon1} = \sum_{i=1}^{d}\frac{\left | P_i - Q_i \right|}{min(P_i,Q_i)}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Vicis_Wave_Hedges)


def wavehedges(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    .. math::

        d_{wh} =\sum_{i=1}^{d}\frac{\left | P_i - Q_i \right |}{max(P_i, Q_i)}

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.Wave_Hedges)


def mpdist(a: ArrayLike, b: ArrayLike, w: int, threshold: float = 0.05) -> ShapeletsArray:
    r"""Calculates the distance between time series using Matrix Profile

    It considers two time series to be similar if they share many similar subsequences, 
    regardless of the order of matching subsequences. MPdist is robust to spikes, warping, 
    linear trends, dropouts, wandering baseline and missing values, issues that are 
    common outside of benchmark datasets.

    References
    ----------
    [1] `Matrix Profile XII: MPdist`_ A Novel Time Series Distance Measure to Allow Data Mining in More Challenging Scenarios.
    *Gharghabi S, Imani S, Bagnall A, Darvishzadeh A, Keogh E.* 2018 IEEE International Conference on Data Mining (ICDM). 2018.

    .. _`Matrix Profile XII: MPdist`: https://sites.google.com/site/mpdistinfo/
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.MPDist, w=w, threshold=threshold)

def dtw(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""Calculates the Dynamic Time Warping Distance.
    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.DTW)

def sbd(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    r"""
    Calculates the Shape-Based distance (SBD).
    
    It computes the normalized cross-correlation and it returns 1.0 minus the value 
    that maximizes the correlation value between each pair of time series.

    """
    return _pygauss.cdist(a, b, _pygauss.DistanceType.SBD)