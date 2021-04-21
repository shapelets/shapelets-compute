from __future__ import annotations

from .__basic_typing import ArrayLike, _ScalarLike
from typing import Optional, Tuple, Union, Literal
from ._array_obj import ShapeletsArray

from . import _pygauss

AnyScalar = _ScalarLike
FloatOrComplex = Union[complex, float]
ScanOp = Literal['add', 'max', 'min', 'mul']

def __map_scan_op(scanop: ScanOp): 
    if scanop == 'add':
        return _pygauss.ScanOp.Add 
    elif scanop == 'max':
        return _pygauss.ScanOp.Max
    elif scanop == 'min':
        return _pygauss.ScanOp.Min
    elif scanop == 'mul':
        return _pygauss.ScanOp.Mul 
    else:
        raise ValueError("Unknown scan operation")

def argmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """

    """
    return _pygauss.argmin(array_like, dim)

def nanargmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """

    """
    return _pygauss.nanargmin(array_like, dim)

def argmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: 
    """
    """
    return _pygauss.argmax(array_like, dim) 

def nanargmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: 
    """
    """
    return _pygauss.nanargmax(array_like, dim) 




def amax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: 
    """

    """
    return _pygauss.amax(array_like, dim)  

def nanmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.nanmax(array_like, dim)  

def maximum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.maximum(left, right) 

def fmax(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.fmax(left, right) 



def count_nonzero(array_like: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.count_nonzero(array_like, dim) 

def sum(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.sum(array_like, dim, nan_value) 

def product(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]:
    """
    """
    return _pygauss.product(array_like, dim, nan_value) 

def cumsum(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.cumsum(array_like, dim) 

def nancumsum(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.nancumsum(array_like, dim) 

def cumprod(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    """
    return _pygauss.cumprod(array_like, dim) 

def nancumprod(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]:
    """
    """
    return _pygauss.nancumprod(array_like, dim)

def scan(array_like: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    """
    """
    return _pygauss.scan(array_like, dim, __map_scan_op(op), inclusive_scan)

def nanscan(array_like: ArrayLike, dim: int = 0, nan: float = 0.0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    """
    """
    return _pygauss.nanscan(array_like, dim, __map_scan_op(op), inclusive_scan)

def diff1(array_like: ArrayLike, dim: int) -> ShapeletsArray:
    """
    """
    return _pygauss.diff1(array_like, dim)

def diff2(array_like: ArrayLike, dim: int) -> ShapeletsArray: 
    """
    """
    return _pygauss.diff2(array_like, dim)

def sort(array_like: ArrayLike, dim: int = 0, asc: bool = True) -> ShapeletsArray: 
    """
    """
    return _pygauss.sort(array_like, dim, asc)

def sort_index(array_like: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.sort_index(array_like, dim, asc)

def sort_by_key(keys: ArrayLike, data: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.sort_by_key(keys, data, dim, asc)

def flatnonzero(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.flatnonzero(array_like)

def union(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    """
    """
    return _pygauss.union(x1, x2, is_unique)

def unique(array_like: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: 
    """
    """
    return _pygauss.unique(array_like, is_sorted)

def intersect(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    """
    """
    return _pygauss.intersect(x1, x2, is_unique)

def count_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.count_by_key(keys, vals, dim)

def max_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.max_by_key(keys, vals, dim)

def min_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.min_by_key(keys, vals, dim)

def product_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.product_by_key(keys, vals, dim)

def nanproduct_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 1.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.nanproduct_by_key(keys, vals, nan_value, dim)

def sum_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.sum_by_key(keys, vals, dim)

def nansum_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 0.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    """
    return _pygauss.nansum_by_key(keys, vals, nan_value, dim)

def scan_by_key(keys: ArrayLike, vals: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray: 
    """
    """
    return _pygauss.scan_by_key(keys, vals, dim, __map_scan_op(op), inclusive_scan)

def nan_to_num(array_like: ArrayLike, nan: float = 0.0, inf: float = 0.0) -> ShapeletsArray:
    """
    Returns a new array where the NaN and Inf values have been replaced.

    Params
    ------
    array_like: ArrayLike
        A valid array expression

    nan: float. Defaults to 0.0
        Value utilized to replace all `nan` values found in ``array_like``

    inf: float: Defaults to 0.0
        Value utilized to replace `inf` values in ``array_like``

    Returns
    -------
    A new array with no NaN or Inf values.

    Examples
    -------

    >>> sc.nan_to_num([[np.NaN, 1.0], [np.Inf, 3.0]], 0.0, 2.0)
    [2 2 1 1]
        0.0000     1.0000 
        2.0000     3.0000     
    """
    return _pygauss.nan_to_num(array_like, nan, inf)

def minimum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.minimum(left, right)

def fmin(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.fmin(left, right) 

def amin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: 
    """
    Returns the minimum of an array or the minimum along an axis, propagating NaNs.

    Parameters
    ----------
    array_like: ArrayLike
    dim: Optioal int.  Defaults to None.

    Returns
    -------
    A scalar value, representing the minimum value of the tensor when no `dim` is specified; otherwise,
    it returns an array, representing the minimum value through a particular axis.

    See Also
    --------
    nanmin: The minimum value of an array along a given axis, ignoring any NaNs.
    minimum: Element-wise minimum of two arrays, propagating any NaNs.
    fmin: Element-wise minimum of two arrays, ignoring any NaNs.
    argmin: Return the indices of the minimum values.
    nanargmin:  Return the indices of the minimum values, ignoring any NaNs.
    """
    return _pygauss.amin(array_like, dim)

def nanmin(x: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    The minimum value of an array along a given axis, ignoring any NaNs.

    Parameters
    ----------
    x: ArrayLike
        Input array

    dim: Optioal int.  Defaults to None.
        Axis along the minimum is computed. 

    Returns
    -------
    Union[FloatOrComplex, ShapeletsArray]
        A scalar value, representing the minimum value of the tensor when no `dim` is specified; otherwise,
        it returns an array, representing the minimum value through a particular axis.

    See Also
    --------
    amin
        The minimum value of an array along a given axis, propagating any NaNs.
    minimum
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin
        Return the indices of the minimum values, propagating any NaNs.
    nanargmin
        Return the indices of the minimum values, ignoring any NaNs.

    """
    return _pygauss.nanmin(x, dim)


    # // amin -> Return the minimum of an array or minimum along an axis, propagating NaNs
    # // nanmin -> The minimum value of an array along a given axis, ignoring any NaNs.
    # // minimum -> Element-wise minimum of two arrays, propagating any NaNs.
    # // fmin -> Element-wise minimum of two arrays, ignoring any NaNs.
    # // argmin -> Return the indices of the minimum values.


def all(array_like: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    """
    Tests if all array elements along a particular axis are `True`

    When dim is not specified, this method will return a boolean flag as it will use all 
    elements in the tensor for the computation.

    Parameters
    ----------
    array_like: ArrayLike
        A valid array expression.

    dim: Optional int, defaults to None
        Dimension or axis where the reduction takes place; when left unspecified, the all tensor 
        elements will participate in the operation

    Returns
    -------
    Union[bool, ShapeletsArray]
        Either a boolean flag or an array of boolean values, depending on the `dim` parameter.

    See Also
    --------
    any
        Tests if any array element along a particular axis is `True`
    all_by_key
        Similar to any, but grouping by keys.    

    Examples
    --------
    >>> a = [[True, True], [False, False], [True, False]]
    >>> sc.all(a)
    False
    >>> sc.all(a, 0)
    [1 2 1 1]
         0          0
    >>> sc.all(a, 1)      
    [3 1 1 1]
         1 
         0 
         0       
    """
    return _pygauss.all(array_like, dim)

def any(array_like: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    """
    Tests if any array element along a particular axis is `True`

    When dim is not specified, this method will return a boolean flag.

    Parameters
    ----------
    array_like: ArrayLike
        A valid array expression.

    dim: Optional int, defaults to None
        Dimension or axis where the reduction takes place; when left unspecified, the all tensor 
        elements will participate in the operation

    Returns
    -------
    Union[bool, ShapeletsArray]
        Either a boolean flag or an array of boolean values, depending on the `dim` parameter.

    See Also
    --------
    all
        Tests if all array elements along a particular axis are `True`

    any_by_key
        Similar to any, but grouping by keys.

    Examples
    --------
    Invoke ``any`` to all elements in the array:
    
    >>> a = [[True, True], [False, False], [True, False]]
    >>> sc.any(a)
    True

    Over the same array, but this time, reduce by rows...

    >>> sc.any(a, 0)
    [1 2 1 1]
         1          1
    
    Or by columns...

    >>> sc.any(a, 1)      
    [3 1 1 1]
         1 
         0 
         1    
    """
    return _pygauss.any(array_like, dim)

def any_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    Reduces the values in `vals` by testing if any of the elements is set to `True`, grouping the results 
    by keys.

    Keys must be a columnar vector compatible with the dimension of the reduction.  Please note that keys types are 
    restricted to signed or usigned ints; if the function is called with any other type, an implicit conversion to 
    signed int will occur automatically.
    
    Keys values may repeat, however only consecutive key values would be considered for each reduction. If a key 
    value is repeated somewhere else in the keys array it will be considered the start of a new reduction.

    Parameters
    ----------
    keys: ArrayLike
        Column vector compatible with the dimension of the operation indicating the elements' group

    vals: ArrayLike
        Tensor with values; if the values are not boolean, zeros will be interpreted as `False`; otherwise, `True`.

    dim: Optional, int.  Defaults to None
        Axis for the operation to ocurr.  

    Returns
    -------
    A tuple of ShapeletsArray.  
        The first element will be the keys and the second element the results of the reduction operation.

    Examples
    --------
    >>> a = sc.array([True, True, False, True])
    >>> k = sc.array([10, 10, 20, 20], dtype="int32")
    >>> r_k, r_d = sc.any_by_key(k, a)
    >>> r_k
    [2 1 1 1]
            10 
            20 
    >>> r_d
    [2 1 1 1]
            1 
            1 
    >>> a = sc.array([[True, True], [False, True]])
    >>> k = sc.array([10,20], dtype="int32")
    >>> r_k, r_d = sc.any_by_key(k, a, 1)
    >>> r_k
    [2 1 1 1]
            10 
            20 
    >>> r_d        
    [2 2 1 1]
            1          1 
            0          1 
    """
    return _pygauss.any_by_key(keys, vals, dim)

def all_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    """
    Reduces the values in `vals` by testing if all of the elements are set to `True`, by grouping the results 
    using the values in `keys`.

    Keys must be a columnar vector compatible with the dimension of the reduction.  Please note that keys types are 
    restricted to signed or usigned ints; if the function is called with any other type, an implicit conversion to 
    signed int will occur automatically.
    
    Keys values may repeat, however only consecutive key values would be considered for each reduction. If a key 
    value is repeated somewhere else in the keys array it will be considered the start of a new reduction.

    Parameters
    ----------
    keys: ArrayLike
        Column vector compatible with the dimension of the operation, indicating the elements' group

    vals: ArrayLike
        Tensor with values; if the values are not boolean, zeros will be interpreted as `False`; otherwise, `True`.

    dim: Optional, int.  Defaults to None
        Axis for the reduction operation.  

    Returns
    -------
    A tuple of ShapeletsArray.  
        The first element will be the keys and the second element the results of the reduction operation.

    Examples
    --------
    >>> a = sc.array([True, True, False, True])
    >>> k = sc.array([10, 10, 20, 20], dtype="int32")
    >>> r_k, r_d = sc.all_by_key(k, a)
    >>> r_k
    [2 1 1 1]
            10 
            20 
    >>> r_d
    [2 1 1 1]
            1 
            0 
    >>> a = sc.array([[True, True], [False, True]])
    >>> k = sc.array([10,20], dtype="int32")
    >>> r_k, r_d = sc.all_by_key(k, a, 1)
    >>> r_k
    [2 1 1 1]
            10 
            20 
    >>> r_d        
    [2 2 1 1]
            1          1 
            0          1 
    """
    return _pygauss.all_by_key(keys, vals, dim)

    

__all__ = [
    "any", "all", "nan_to_num", "amin", "nanmin", "minimum", "fmin", "argmin", "nanargmin", "amax", "nanmax", "maximum", 
    "fmax", "argmax", "nanargmax", "count_nonzero", "sum", "product", "cumsum", "nancumsum", "cumprod", "nancumprod", 
    "scan", "nanscan", "diff1", "diff2", "sort", "sort_index", "sort_by_key", "flatnonzero", "unique", "union", "intersect", 
    "any_by_key", "all_by_key", "count_by_key", "max_by_key", "min_by_key", "product_by_key", "nanproduct_by_key", 
    "sum_by_key", "nansum_by_key", "scan_by_key"
]