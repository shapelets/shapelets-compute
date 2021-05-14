from __future__ import annotations

from .__basic_typing import ArrayLike, _ScalarLike
from typing import Optional, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ._array_obj import ShapeletsArray

from . import _pygauss

AnyScalar = _ScalarLike
FloatOrComplex = Union[float, complex]

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

def argmin(a: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    r"""
    Locates the minimum values in an array

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Tuple[int, scalar]
        When no dimension is given, it returns a tuple with the index and value
    Tuple[ShapeletsArray, ShapeletsArray]
        When the operation occurs over a concrete axis, it returns two arrays, one for the index and one for the values.    

    Notes
    -----
    When the operation does not indicate an axis, the index refers to the position of 
    the element in column major order.  When an axis is given, the index refers to the 
    position of the element within the axis.

    See also
    --------
    nanargmin

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3],[4,5,6]])
    >>> sc.argmin(a)
    (0, 1.0)
    >>> indices, values = sc.argmin(a, 0)
    >>> indices
    [1 3 1 1]
        0          0          0 
    >>> values
    [1 3 1 1]
        1          2          3              
    """  
    return _pygauss.argmin(a, dim)

def nanargmin(a: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]:
    r"""
    Locates the minimum values in an array, discarding nan values

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Tuple[int, scalar]
        When no dimension is given, it returns a tuple with the index and value
    Tuple[ShapeletsArray, ShapeletsArray]
        When the operation occurs over a concrete axis, it returns two arrays, one for the index and one for the values.    

    Notes
    -----
    When the operation does not indicate an axis, the index refers to the position of 
    the element in column major order.  When an axis is given, the index refers to the 
    position of the element within the axis.

    See also
    --------
    argmin

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([[np.nan, 2, 3],[4, 5, np.nan]])
    >>> sc.nanargmin(a)
    (2, 2.0)
    >>> indices, values = sc.nanargmin(a, 0)
    >>> indices
    [1 3 1 1]
        1          0          0 
    >>> values
    [1 3 1 1]
        4.0000     2.0000     3.0000               
    """
    return _pygauss.nanargmin(a, dim)

def argmax(a: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: 
    r"""
    Locates the maximum values in an array

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Tuple[int, scalar]
        When no dimension is given, it returns a tuple with the index and value
    Tuple[ShapeletsArray, ShapeletsArray]
        When the operation occurs over a concrete axis, it returns two arrays, one for the index and one for the values.    

    Notes
    -----
    When the operation does not indicate an axis, the index refers to the position of 
    the element in column major order.  When an axis is given, the index refers to the 
    position of the element within the axis.

    See also
    --------
    nanargmax

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3],[4,5,6]])
    >>> sc.argmax(a)
    (5, 6.0)
    >>> indices, values = sc.argmax(a, 0)
    >>> indices
    [1 3 1 1]
        1          1          1 
    >>> values
    [1 3 1 1]
        4          5          6    
    """
    return _pygauss.argmax(a, dim) 

def nanargmax(a: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: 
    r"""
    Locates the maximum values in an array, discarding nan values

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Tuple[int, scalar]
        When no dimension is given, it returns a tuple with the index and value
    Tuple[ShapeletsArray, ShapeletsArray]
        When the operation occurs over a concrete axis, it returns two arrays, one for the index and one for the values.    

    Notes
    -----
    When the operation does not indicate an axis, the index refers to the position of 
    the element in column major order.  When an axis is given, the index refers to the 
    position of the element within the axis.

    See also
    --------
    argmax

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([[np.nan, 2, 3],[4, 5, np.nan]])
    >>> sc.nanargmax(a)
    (3, 5.0)
    >>> indices, values = sc.nanargmax(a, 0)
    >>> indices
    [1 3 1 1]
        1          1          0 
    >>> values
    [1 3 1 1]
        4.0000     5.0000     3.0000 
    """
    return _pygauss.nanargmax(a, dim) 

def amax(a: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: 
    r"""
    Return the maximum of an array or maximum along an axis

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Scalar
        When no dimension is given, it returns a value
    ShapeletsArray
        When the operation occurs over a concrete axis, it returns an array with the minimum across 
        the dimensionality of the selected axis. 

    See also
    --------
    nanmax

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3],[4,5,6]])
    >>> sc.amax(a)
    6.0
    >>> sc.amax(a, 0)
    [1 3 1 1]
        4          5          6 
    """
    return _pygauss.amax(a, dim)  

def nanmax(a: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Return the maximum of an array or maximum along an axis, ignoring any NaNs.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. 
    
    Returns
    -------
    Scalar
        When no dimension is given, it returns a value
    ShapeletsArray
        When the operation occurs over a concrete axis, it returns an array with the minimum across 
        the dimensionality of the selected axis. 

    See also
    --------
    amax

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([[np.nan, 2, 3],[4, 5, np.nan]])
    >>> sc.nanmax(a)
    5.0
    >>> sc.nanmax(a, 0)
    [1 3 1 1]
        4.0000     5.0000     3.0000  
    """
    return _pygauss.nanmax(a, dim)  

def maximum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise maximum of array elements, propagating NaNs.

    Parameters
    ----------
    left: ArrayLike
        Input array expression
    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the maximum of left and right, element-wise.

    Notes
    -----
    At least one of the entries must be an array expression.  Arrays must be of 
    the same shape as no broadcasting rules are applied to this function; use 
    :obj:`~shapelets.compute.tile` method for an efficient mechanism to tile 
    the arrays.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np  
    >>> sc.maximum([1,2,3], [-1, np.nan, 10])
    [3 1 1 1]
         1.0000 
         2.0000 
        10.0000
    """
    return _pygauss.maximum(left, right) 

def fmax(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.fmax(left, right) 

def count_nonzero(a: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.count_nonzero(a, dim) 

def sum(a: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.sum(a, dim, nan_value) 

def product(a: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]:
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.product(a, dim, nan_value) 

def cumsum(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.cumsum(a, dim) 

def nancumsum(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.nancumsum(a, dim) 

def cumprod(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.cumprod(a, dim) 

def nancumprod(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]:
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.nancumprod(a, dim)

def scan(a: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.scan(a, dim, __map_scan_op(op), inclusive_scan)

def nanscan(a: ArrayLike, dim: int = 0, nan: float = 0.0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc

    """
    return _pygauss.nanscan(a, dim, __map_scan_op(op), inclusive_scan)

def diff1(a: ArrayLike, dim: int) -> ShapeletsArray:
    r"""
    First order numerical difference along specified dimension

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc

    """
    return _pygauss.diff1(a, dim)

def diff2(a: ArrayLike, dim: int) -> ShapeletsArray: 
    r"""
    Second order numerical difference along specified dimension

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc

    """
    return _pygauss.diff2(a, dim)

def sort(a: ArrayLike, dim: int = 0, asc: bool = True) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.sort(a, dim, asc)

def sort_index(a: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.sort_index(a, dim, asc)

def sort_by_key(keys: ArrayLike, data: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc

    """
    return _pygauss.sort_by_key(keys, data, dim, asc)

def flatnonzero(a: ArrayLike) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.flatnonzero(a)

def union(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.union(x1, x2, is_unique)

def unique(a: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.unique(a, is_sorted)

def intersect(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.intersect(x1, x2, is_unique)

def count_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.count_by_key(keys, vals, dim)

def max_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.max_by_key(keys, vals, dim)

def min_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.min_by_key(keys, vals, dim)

def product_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.product_by_key(keys, vals, dim)

def nanproduct_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 1.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.nanproduct_by_key(keys, vals, nan_value, dim)

def sum_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.sum_by_key(keys, vals, dim)

def nansum_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 0.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.nansum_by_key(keys, vals, nan_value, dim)

def scan_by_key(keys: ArrayLike, vals: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.scan_by_key(keys, vals, dim, __map_scan_op(op), inclusive_scan)

def nan_to_num(a: ArrayLike, nan: float = 0.0, inf: float = 0.0) -> ShapeletsArray:
    r"""
    Returns a new array where the NaN and Inf values have been replaced.

    Params
    ------
    a: ArrayLike
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
    return _pygauss.nan_to_num(a, nan, inf)

def minimum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc    
    """
    return _pygauss.minimum(left, right)

def fmin(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> import shapelets.compute as sc

    """
    return _pygauss.fmin(left, right) 

def amin(a: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: 
    r"""
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
    return _pygauss.amin(a, dim)

def nanmin(a: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    The minimum value of an array along a given axis, ignoring any NaNs.

    Parameters
    ----------
    a: ArrayLike
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
    return _pygauss.nanmin(a, dim)


    # // amin -> Return the minimum of an array or minimum along an axis, propagating NaNs
    # // nanmin -> The minimum value of an array along a given axis, ignoring any NaNs.
    # // minimum -> Element-wise minimum of two arrays, propagating any NaNs.
    # // fmin -> Element-wise minimum of two arrays, ignoring any NaNs.
    # // argmin -> Return the indices of the minimum values.


def all(a: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    r"""
    Tests if all array elements along a particular axis are `True`

    When dim is not specified, this method will return a boolean flag as it will use all 
    elements in the tensor for the computation.

    Parameters
    ----------
    a: ArrayLike
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
    return _pygauss.all(a, dim)

def any(a: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    r"""
    Tests if any array element along a particular axis is `True`

    When dim is not specified, this method will return a boolean flag.

    Parameters
    ----------
    a: ArrayLike
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
    return _pygauss.any(a, dim)

def any_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
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
    r"""
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