
from typing import Optional, Union, Literal
from .statistics import (mean as __mean, std as __std, var as __var, topk_max, topk_min, quantile as __quantile, median as __median, 
                         covariance as __cov, correlation as __corr, cross_correlation as __cross_cor)
from ._extract_transform import (join as __join)
from .__basic_typing import ArrayLike, DataTypeLike
from ._array_obj import ShapeletsArray, array as asarray


def average(a: ArrayLike, axis: Optional[int] = None, weights = Optional[ArrayLike]) -> Union[float, complex, ShapeletsArray]:
    return __mean(a, weights, axis)

def mean(a: ArrayLike, axis: Optional[int] = None) -> Union[float, complex, ShapeletsArray]: 
    return __mean(a, None, axis)

def std(a: ArrayLike, axis: Optional[int] = None, ddof: int = 0) -> Union[float, complex, ShapeletsArray]:
    return __std(a, axis)

def var(a: ArrayLike, axis: Optional[int] = None, dtype: Optional[DataTypeLike] = None, ddof: int = 0) -> Union[float, complex, ShapeletsArray]:
    arr = asarray(a)
    if not arr.is_floating and dtype is not None:
        arr = arr.astype(dtype)
    return __var(arr, None, axis, ddof != 0)

def percentile(a: ArrayLike, q: ArrayLike, axis: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    pass

def quantile(a: ArrayLike, q: ArrayLike, axis: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    pass

def median(a: ArrayLike, axis: Optional[int] = None)-> Union[float, complex, ShapeletsArray]:
    pass

def corrcoef(x: ArrayLike, y: Optional[ArrayLike] = None, rowvar: bool = True) -> Union[float, complex, ShapeletsArray]:
    arr = __join([x, y], 0 if rowvar else 1) if y is not None else asarray(x) 

    if rowvar:
        arr = arr.T 
    res = __corr(arr)
    if rowvar:
        res = res.T 
    return res

def correlate(a: ArrayLike, v: ArrayLike, mode: Optional[Literal['valid', 'same', 'full']] = None) -> ShapeletsArray:
    return __cross_cor(a, v)

def cov(m: ArrayLike, y: Optional[ArrayLike] = None, rowvar: bool = True, bias: bool = False) -> ShapeletsArray:
    arr = __join([m, y], 0 if rowvar else 1) if y is not None else asarray(m)
    if rowvar:
        arr = arr.T 
    res = __cov(arr, not bias)
    if rowvar:
        res = res.T 
    return res


__all__ = [
    "average", "mean", "std", "var", "percentile", "quantile", "median", "corrcoef", "correlate", "cov", "topk_min", "topk_max"
]






