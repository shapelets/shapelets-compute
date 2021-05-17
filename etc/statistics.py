# %%

# https://github.com/vojd/octave-signal-1.3.2/blob/2f7976de48/inst/xcorr.m
# https://lists.gnu.org/archive/html/help-octave/1995-05/msg00027.html

import shapelets.compute as sc 
import numpy as np
import matplotlib.pyplot as plt
from shapelets.compute.statistics import mean 
np.set_printoptions(precision=4, suppress=True)

x = sc.linspace(-2*np.pi, 2*np.pi, 10, dtype="float64")
y1 = sc.sin(x) 
y2 = sc.cos(x) 
y3 = y1*y2
y4 = x*x -(x/2.0) + 3.0
z = sc.join([y1,y2], 0)
yy = sc.join([y1, y2, y3, y4], 1)
yynp = np.array(yy)

# %%
def xcor(x, y, maxlag = None, scale = None):
    n = max(x.shape[0], y.shape[0])
    maxlag = n-1 if maxlag is None else maxlag
    scale = 'none' if scale is None else scale 

    ysso = sc.flip(y, 0)
    ysso = sc.reorder(ysso, 0, 2, 1, 3)
    r = sc.convolve1(x, ysso, sc.ConvMode.Expand, sc.ConvDomain.Frequency) 

    # so it matches octave
    r = sc.flip(r, 0)

    if scale=='biased':
        r = r / n 
    elif scale=='unbiased':
        adj = sc.join([sc.arange(1, n+1, 1), sc.arange(n-1, 0, -1)], 0)
        adj_mat = sc.tile(adj, 1, r.shape[1], r.shape[2])
        r = r / adj_mat
    elif scale=='coeff':
        rms_x = sc.sum(sc.power(x, 2.0), 0)
        rms_y = sc.sum(sc.power(y, 2.0), 0)
        rms_mat = sc.matmulTN(rms_x, rms_y)
        rms_mat = sc.reorder(rms_mat, 2, 0, 1, 3)
        rms_mat = sc.tile(rms_mat, r.shape[0])
        r = r / sc.sqrt(rms_mat)

    if (maxlag < n-1):
        r = r[n-maxlag-1:maxlag-n-1,...]

    return (sc.arange(-maxlag, maxlag+1, 1), r)

def xcov(x, y, maxlag = None, scale = None):
    """
    Computes the cross-covariance between all columns in x 
    and all columns in y.

    This method is equivalent to compute the cross-correlation but,
    in the case of cross-covariance, the means of each column are 
    substracted before computing the cross-correlation.

    Parameters
    ----------
    x: Columnar matrix.
    y: Columnar matrix.
    maxlag: int, defaults to None
    When not specified, it will return all factors, that is, N-1, where
    N is the greater of the lengths of x and y.
    scale: literal, defaults to None or 'none'
    See xcorr function.
    """

    meanXss = sc.mean(x, 0)
    meanYss = sc.mean(y, 0)
    xsso = x - sc.tile(meanXss, x.shape[0])
    ysso = y - sc.tile(meanYss, y.shape[0])
    return xcor(xsso, ysso, maxlag=maxlag, scale=scale)


def autocor(x, maxlag = None, scale = None):
    tmp = xcor(x, x, maxlag, scale)[1]
    lst = [tmp[:, ii, ii, :] for ii in range(x.shape[1])]
    return sc.join(lst, 1)

def autocov(x, maxlag = None, scale = None):
    tmp = xcov(x,x,maxlag, scale)[1]
    lst = [tmp[:, ii, ii, :] for ii in range(x.shape[1])]
    return sc.join(lst, 1)

# %%
def cov(x, ddof=1):
    fact = x.shape[0] - ddof
    mean_x = sc.mean(x, 0)
    xsso = x - sc.tile(mean_x, x.shape[0])
    return sc.matmulTN(xsso, sc.conjugate(xsso)) / fact

# %%
def corrcoef(x, ddof=1):
    covm = cov(x, ddof)
    diag = sc.diag(covm, 0, True)
    factor = sc.matmulNT(diag, diag)
    sq_factor = sc.sqrt(factor)
    return covm / sq_factor

# %%
