from typing import Optional, Tuple, Union

from . import _pygauss
from .__basic_typing import ArrayLike, _ScalarLike
from ._array_obj import ShapeletsArray

AnyScalar = _ScalarLike
FloatOrComplex = Union[complex, float]

class ScanOp:
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...
    Add: _pygauss.ScanOp # value = <ScanOp.Add: 0>
    Max: _pygauss.ScanOp # value = <ScanOp.Max: 3>
    Min: _pygauss.ScanOp # value = <ScanOp.Min: 2>
    Mul: _pygauss.ScanOp # value = <ScanOp.Mul: 1>
    __members__: dict # value = {'Add': <ScanOp.Add: 0>, 'Mul': <ScanOp.Mul: 1>, 'Min': <ScanOp.Min: 2>, 'Max': <ScanOp.Max: 3>}


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
    Either a boolean flag or an array of boolean values, depending on the `dim` parameter.

    See Also
    --------
    all: Tests if all array elements along a particular axis are `True`
    any_by_key: Similar to any, but grouping by keys.

    Examples
    --------
    >>> a = [[True, True], [False, False], [True, False]]
    >>> sc.any(a)
    True
    >>> sc.any(a, 0)
    [1 2 1 1]
         1          1
    >>> sc.any(a, 1)      
    [3 1 1 1]
         1 
         0 
         1    
    """

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
    Either a boolean flag or an array of boolean values, depending on the `dim` parameter.

    See Also
    --------
    any: Tests if any array element along a particular axis is `True`
    all_by_key: Similar to any, but grouping by keys.    

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
    dim: Optiona, int.  Defaults to None
    Axis for the operation to ocurr.  

    Returns
    -------
    A tuple of ShapeletsArray.  The first element will be the keys and the second element the results of the 
    reduction operation.

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
    A tuple of ShapeletsArray.  The first element will be the keys and the second element the results of the 
    reduction operation.

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

def nan_to_num(array_like: ArrayLike, nan: float = 0.0, inf: float = 0.0) -> ShapeletsArray:
    """
    Returns a new array where the NaN and Inf values have been replaced.

    Params
    ------
    array_like: ArrayLike
    nan: float. Defaults to 0.0
    info: float: Defaults to 0.0

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

    # // amin -> Return the minimum of an array or minimum along an axis, propagating NaNs
    # // nanmin -> The minimum value of an array along a given axis, ignoring any NaNs.
    # // minimum -> Element-wise minimum of two arrays, propagating any NaNs.
    # // fmin -> Element-wise minimum of two arrays, ignoring any NaNs.
    # // argmin -> Return the indices of the minimum values.

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

def nanmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    """
    The minimum value of an array along a given axis, ignoring any NaNs.

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
    amin: The minimum value of an array along a given axis, propagating any NaNs.
    minimum: Element-wise minimum of two arrays, propagating any NaNs.
    fmin: Element-wise minimum of two arrays, ignoring any NaNs.
    argmin: Return the indices of the minimum values, propagating any NaNs.
    nanargmin:  Return the indices of the minimum values, ignoring any NaNs.
    """

def argmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]: ... 
def nanargmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: ...
def minimum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: ...
def fmin(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: ...
def amax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: ...
def nanmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: ...
def maximum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: ...
def fmax(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: ...
def argmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: ...
def nanargmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, FloatOrComplex], Tuple[ShapeletsArray, ShapeletsArray]]: ...
def count_nonzero(array_like: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: ...

def sum(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]: ...
def product(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]: ...
def cumsum(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: ...
def nancumsum(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: ...
def cumprod(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: ...
def nancumprod(array_like: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: ...
def scan(array_like: ArrayLike, dim: int = 0, op: ScanOp = ScanOp.Add, inclusive_scan: bool = True) -> ShapeletsArray: ...
def nanscan(array_like: ArrayLike, dim: int = 0, nan: float = 0.0, op: ScanOp = ScanOp.Add, inclusive_scan: bool = True) -> ShapeletsArray: ...
def diff1(array_like: ArrayLike, dim: int) -> ShapeletsArray: ...

def diff2(array_like: ArrayLike, dim: int) -> ShapeletsArray: ...
def sort(array_like: ArrayLike, dim: int = 0, asc: bool = True) -> ShapeletsArray: ...
def sort_index(array_like: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def sort_by_key(keys: ArrayLike, data: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def flatnonzero(array_like: ArrayLike) -> ShapeletsArray: ...
def union(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: ...
def unique(array_like: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: ...
def intersect(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: ...

def count_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def max_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def min_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def product_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def nanproduct_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 1.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def sum_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def nansum_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 0.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: ...
def scan_by_key(keys: ArrayLike, vals: ArrayLike, dim: int = 0, op: ScanOp = ScanOp.Add, inclusive_scan: bool = True) -> ShapeletsArray: ...
