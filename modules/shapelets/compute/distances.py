from typing import Callable, TypedDict
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    DistanceType,
    pdist as _pdist,
    cdist as _cdist
    
)


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
    return _pdist(tss, metric, **kwargs)

def cdist(xa: ArrayLike, xb: ArrayLike, metric: DistanceType, **kwargs) -> ShapeletsArray: 
    """
    Compute distance between each pair of the two collections of inputs.
    """
    return _cdist(xa, xb, metric, **kwargs)


def abs_euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Abs_Euclidean)

def additive_symm_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Additive_Symm_Chi)

def avg_l1_linf(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Avg_L1_Linf)

def bhattacharyya(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Bhattacharyya)

def canberra(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Canberra)

def chebyshev(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Chebyshev)

def clark(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Clark)

def cosine(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Cosine)

def czekanowski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Czekanowski)

def dice(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Dice)

def divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Divergence)

def dtw(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.DTW)

def euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Euclidean)

def fidelity(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Fidelity)

def gower(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Gower)

def hamming(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Hamming)

def harmonic_mean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Harmonic_mean)

def hellinger(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Hellinger)

def innerproduct(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Innerproduct)

def intersection(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Intersection)

def jaccard(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Jaccard)

def jeffrey(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Jeffrey)

def jensen_difference(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Jensen_Difference)

def jensen_shannon(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Jensen_Shannon)

def k_divergence(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.K_Divergence)

def kulczynski(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Kulczynski)

def kullback(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Kullback)

def kumar_johnson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Kumar_Johnson)

def kumarhassebrook(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Kumar_Hassebrook)

def lorentzian(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Lorentzian)

def manhattan(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Manhattan)

def matusita(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Matusita)

def max_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Max_Symmetric_Chi)

def min_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Min_Symmetric_Chi)

def minkowshi(a: ArrayLike, b: ArrayLike, p: float) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Minkowshi)

def neyman(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Neyman)

def pearson(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Pearson)

def prob_symmetric_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Prob_Symmetric_Chi)

def sbd(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.SBD)

def soergel(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Soergel)

def sorensen(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Sorensen)

def square_chord(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Square_Chord)

def squared_chi(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Squared_Chi)

def squared_euclidean(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Squared_Euclidean)

def taneja(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Taneja)

def topsoe(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Topsoe)

def vicis_wave_hedges(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Vicis_Wave_Hedges)

def wavehedges(a: ArrayLike, b: ArrayLike) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Wave_Hedges)


def mpdist(a: ArrayLike, b: ArrayLike, w: int, threshold: float = 0.05) -> ShapeletsArray: 
    return cdist(a, b, DistanceType.MPDist, w=w, threshold=threshold)
    
def minkowshi(a: ArrayLike, b: ArrayLike, p: float) -> ShapeletsArray:
    return cdist(a, b, DistanceType.Minkowshi, p=p)


