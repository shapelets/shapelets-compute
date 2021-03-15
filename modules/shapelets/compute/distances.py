from typing import Literal
from ._pygauss import (
    euclidian as _euclidian, 
    hamming as _hamming, 
    manhattan as _manhattan, 
    sbd as _sbd, 
    dtw as _dtw
)   

MetricType = Literal["euclidian", "hamming", "manhattan", "sbd", "dtw", "mpdist"]

def pdist(tss, metric: MetricType): 
    """
    Pairwise distances between observations in n-dimensional space.
    """
    pass

def cdist(xa, xb, metric: MetricType):
    """
    Compute distance between each pair of the two collections of inputs.
    """
    pass 

def euclidian(a, b):
    """
    Computes the Euclidean distance between two 1-D arrays.
    """
    return cdist(a, b, "euclidian")

def hamming(a, b):
    """
    Computes the Hamming distance between two 1-D arrays.
    """
    return cdist(a, b, "hamming")

def cityblock(a, b):
    """
    Computes the City Block (a.k.a. Manhantan) distance between two 1-D arrays.
    """
    return cdist(a, b, "manhatan")

def sbd(a, b):
    """
    Computes the Shape-Based distance (SBD) between two 1-D arrays.
    
    It computes the normalized cross-correlation and it returns 1.0 minus the value 
    that maximizes the correlation value between each pair of time series.
    """
    return cdist(a, b, "sbd")

def dtw(a, b):
    """
    Computes the Dynamic Time Warping Distance between two 1-D arrays.
    """
    return cdist(a, b, "dtw")

def mpdist(a, b):
    """
    Computes the Matrix Profile Distance between two 1-D arrays.
    """
    return cdist(a, b, "mpdist")    
    