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
    Return the maximum of an array or along an axis

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
        When the operation occurs over a concrete axis, it returns an array with the maximum across 
        the dimensionality of the selected axis. 

    See Also
    --------
    nanmax
        The minimum value of an array along a given axis, ignoring any NaNs.
    maximum
        Element-wise minimum of two arrays, propagating any NaNs.
    fmax
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmax
        Return the indices of the minimum values.
    nanargmax
        Return the indices of the minimum values, ignoring any NaNs.

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

def amin(a: ArrayLike, dim: Optional[int] = None) -> Union[AnyScalar, ShapeletsArray]: 
    r"""
    Return the minimum of an array or along an axis

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

    See Also
    --------
    nanmin
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin
        Return the indices of the minimum values.
    nanargmin
        Return the indices of the minimum values, ignoring any NaNs.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3],[4,5,6]])
    >>> sc.amin(a)
    1.0
    >>> sc.amin(a, 0)
    [1 3 1 1]
        1          2          3 

    """
    return _pygauss.amin(a, dim)    

def nanmax(a: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Return the maximum of an array or along an axis, ignoring any NaNs.

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
        When the operation occurs over a concrete axis, it returns an array with the maximum across 
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

def nanmin(a: ArrayLike, dim: Optional[int] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Return the minimum of an array or along an axis, ignoring any NaNs.

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
    amin

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([[np.nan, 2, 3],[4, 5, np.nan]])
    >>> sc.nanmin(a)
    2.0
    >>> sc.nanmin(a, 0)
    [1 3 1 1]
        4.0000     2.0000     3.0000   
    """
    return _pygauss.nanmin(a, dim)


    # // amin -> Return the minimum of an array or minimum along an axis, propagating NaNs
    # // nanmin -> The minimum value of an array along a given axis, ignoring any NaNs.
    # // minimum -> Element-wise minimum of two arrays, propagating any NaNs.
    # // fmin -> Element-wise minimum of two arrays, ignoring any NaNs.
    # // argmin -> Return the indices of the minimum values.

def maximum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise maximum of array elements, respecting Nan values

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
    At least one of the entries must be an array expression.  Broadcasting rules 
    apply to this function.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np  
    >>> sc.maximum([1,2,3], [-1, np.nan, 10])
    [3 1 1 1]
         1.0000 
            nan 
        10.0000  
    >>> sc.maximum(1, [-1, np.nan, 10])
    [3 1 1 1]
         1.0000 
            nan 
        10.0000 
    >>> a = sc.array([1.,2,3])
    >>> sc.maximum(a.T, [-1, np.nan, 10])
    [3 3 1 1]
        1.0000     2.0000     3.0000 
           nan        nan        nan 
       10.0000    10.0000    10.0000        
    """
    return _pygauss.maximum(left, right) 

def minimum(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    r"""
    Element-wise minimum of array elements, respecting Nan values

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the minimum of left and right, element-wise.

    Notes
    -----
    At least one of the entries must be an array expression.  Broadcasting rules 
    apply to this function.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np  
    >>> sc.minimum([1,2,3], [-1, np.nan, 10])
    [3 1 1 1]
       -1.0000 
           nan 
        3.0000  
    >>> sc.minimum(1, [-1, np.nan, 10])
    [3 1 1 1]
       -1.0000 
           nan 
        1.0000 
    >>> a = sc.array([1.,2,3])
    >>> sc.minimum(a.T, [-1, np.nan, 10])
    [3 3 1 1]
       -1.0000    -1.0000    -1.0000 
           nan        nan        nan 
        1.0000     2.0000     3.0000 
    """    
    return _pygauss.minimum(left, right)

def fmax(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise maximum of array elements, ignoring Nan values

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
    At least one of the entries must be an array expression.  Broadcasting rules 
    apply to this function.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np  
    >>> sc.fmax([1,2,3], [-1, np.nan, 10])
    [3 1 1 1]
         1.0000 
         2.0000 
        10.0000  
    >>> sc.fmax(1, [-1, np.nan, 10])
    [3 1 1 1]
         1.0000 
         1.0000 
        10.0000 
    >>> a = sc.array([1.,2,3])
    >>> sc.fmax(a.T, [-1, np.nan, 10])
    [3 3 1 1]
        1.0000     2.0000     3.0000 
        1.0000     2.0000     3.0000 
       10.0000    10.0000    10.0000     
    """
    return _pygauss.fmax(left, right) 

def fmin(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise minimum of array elements, ignoring Nan values

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the minimum of left and right, element-wise.

    Notes
    -----
    At least one of the entries must be an array expression.  Broadcasting rules 
    apply to this function.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np  
    >>> sc.fmin([1.,2,3], [-1, np.nan, 10])
    [3 1 1 1]
       -1.0000 
        2.0000 
        3.0000 
    >>> sc.fmin(1, [-1, np.nan, 10])
    [3 1 1 1]
       -1.0000 
        1.0000 
        1.0000 
    >>> a = sc.array([1.,2,3])
    >>> sc.fmin(a.T, [-1, np.nan, 10])
    [3 3 1 1]
       -1.0000    -1.0000    -1.0000 
        1.0000     2.0000     3.0000 
        1.0000     2.0000     3.0000   
    """
    return _pygauss.fmin(left, right) 

def count_nonzero(a: ArrayLike, dim: Optional[int] = None) -> Union[int, ShapeletsArray]: 
    r"""
    Counts the number of non-zero values in the array a.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. If left unspecified, it will reduce the output of the operation to an scalar value.
    
    Returns
    -------
    Union
        Either a scalar value, if no dimension is specified, or a new array instance.

    Examples
    --------
    >>> import shapelets.compute as sc   
    >>> sc.count_nonzero([1, 0, 3, 0, 5])
    3
    >>> sc.count_nonzero([[1,0], [3,0]], 0)
    [1 2 1 1]
         2          0 
    """
    return _pygauss.count_nonzero(a, dim) 

def sum(a: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Sums values of an input array along an optional dimension.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. If left unspecified, it will reduce the output of the operation to an scalar value.
    
    nan_value: Optional float (default: None)
        Value to replace occurrences of NaN. 

    Returns
    -------
    Union
        Either a scalar value, if no dimension is specified, or a new array instance.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> sc.sum([1, 2, 3, 4, 5])
    15.0
    >>> sc.sum([[1,2], [3,4]], 0)
    [1 2 1 1]
         4          6   

    >>> import shapelets.compute as sc    
    """
    return _pygauss.sum(a, dim, nan_value) 

def product(a: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[FloatOrComplex, ShapeletsArray]:
    r"""
    Computes the product of values along an optional dimension.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: Optional int (default: None)
        Axis of the operation. If left unspecified, it will reduce the output of the operation to an scalar value.
    
    nan_value: Optional float (default: None)
        Value to replace occurrences of NaN.  

    Returns
    -------
    Union
        Either a scalar value, if no dimension is specified, or a new array instance.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> sc.product([1, 2, 3, 4, 5])
    120.0
    >>> sc.product([[1,2], [3,4]], 0)
    [1 2 1 1]
        3          8     
    """
    return _pygauss.product(a, dim, nan_value) 

def cumsum(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 
    
    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> sc.cumsum(a)
    [3 3 1 1]
         1          2          3 
         5          7          9 
        12         15         18     
    >>> sc.cumsum(a, 1)
    [3 3 1 1]
         1          3          6 
         4          9         15 
         7         15         24     
    """
    return _pygauss.cumsum(a, dim) 

def nancumsum(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Cumulative sum over a given axis treating NaNs as zero

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 
    
    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> import numpy as np
    >>> a = sc.array([[1,np.nan,3],[4,5,np.nan],[np.nan,8,9]])
    >>> sc.nancumsum(a)       
    [3 3 1 1]
        1.0000     0.0000     3.0000 
        5.0000     5.0000     3.0000 
        5.0000    13.0000    12.0000     
    >>> sc.nancumsum(a, 1)
    [3 3 1 1]
        1.0000     1.0000     4.0000 
        4.0000     9.0000     9.0000 
        0.0000     8.0000    17.0000     
    """
    return _pygauss.nancumsum(a, dim) 

def cumprod(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]: 
    r"""
    Cumulative product of elements along a given axis

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 
    
    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> sc.cumprod(a)   
    [3 3 1 1]
         1          2          3 
         4         10         18 
        28         80        162      
    >>> sc.cumprod(a, 1)
    [3 3 1 1]
         1          2          6 
         4         20        120 
         7         56        504     
    """
    return _pygauss.cumprod(a, dim) 

def nancumprod(a: ArrayLike, dim: int = 0) -> Union[FloatOrComplex, ShapeletsArray]:
    r"""
    Cumulative sum over a given axis treating NaNs as one

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 
    
    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> import numpy as np
    >>> a = sc.array([[1,np.nan,3],[4,5,np.nan],[np.nan,8,9]])
    >>> sc.nancumprod(a)       
    [3 3 1 1]
        1.0000     1.0000     3.0000 
        4.0000     5.0000     3.0000 
        4.0000    40.0000    27.0000  
    >>> sc.nancumprod(a, 1)
    [3 3 1 1]
        1.0000     1.0000     3.0000 
        4.0000    20.0000    20.0000 
        1.0000     8.0000    72.0000    
    """
    return _pygauss.nancumprod(a, dim)

def scan(a: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    r"""
    Generalization of cummulative operations over a particular axis

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    op: ScanOp (default: 'add')
        Operation to perform.  It could be one of 'add', 'max', 'min' or 'mul'.
    
    inclusive_scan: bool (default: True)
        Indicates if the scan operation is inclusive or exclusive.  In inclusive mode, the first values through 
        an axis will be kept in the result; on the other hand, exclusive scans will introduce Inf, zeros or one 
        values, depending on the operation selected.

    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    See also
    --------
    nanscan 
        For a version resilient to NaN values.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]], dtype="float32")
    >>> sc.scan(a, 0, 'add', False)
    [3 3 1 1]
        0.0000     0.0000     0.0000 
        1.0000     2.0000     3.0000 
        5.0000     7.0000     9.0000     
    >>> sc.scan(a, 0, 'add', True)
    [3 3 1 1]
        1.0000     2.0000     3.0000 
        5.0000     7.0000     9.0000 
       12.0000    15.0000    18.0000             
    """
    return _pygauss.scan(a, dim, __map_scan_op(op), inclusive_scan)

def nanscan(a: ArrayLike, dim: int = 0, nan: float = 0.0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray:
    r"""
    Generalization of cummulative operations over a particular axis

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    nan: float (default: 0)
        Value to use instead of NaN

    op: ScanOp (default: 'add')
        Operation to perform.  It could be one of 'add', 'max', 'min' or 'mul'.
    
    inclusive_scan: bool (default: True)
        Indicates if the scan operation is inclusive or exclusive.  In inclusive mode, the first values through 
        an axis will be kept in the result; on the other hand, exclusive scans will introduce Inf, zeros or one 
        values, depending on the operation selected.

    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([[np.nan,2,3],[4,np.nan,6],[7,8,np.nan]])
    >>> sc.nanscan(a, 0, 0.0, 'add', False)
    [3 3 1 1]
        0.0000     0.0000     0.0000 
        0.0000     2.0000     3.0000 
        4.0000     2.0000     9.0000  
    >>> sc.nanscan(a, 0, 0.0, 'add', True)
    [3 3 1 1]
         0.0000     2.0000     3.0000 
         4.0000     2.0000     9.0000 
        11.0000    10.0000     9.0000     
    """
    return _pygauss.nanscan(a, dim, nan, __map_scan_op(op), inclusive_scan)

def diff1(a: ArrayLike, dim: int = 0) -> ShapeletsArray:
    r"""
    First order numerical difference along specified dimension.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array, except 
        in the dimension 'dim', which will have one element less.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> sc.diff1(a, 0)
    [2 3 1 1]
        3          3          3 
        3          3          3 
    >>> sc.diff1(a, 1)
    [3 2 1 1]
        1          1 
        1          1 
        1          1     
    """
    return _pygauss.diff1(a, dim)

def diff2(a: ArrayLike, dim: int = 0) -> ShapeletsArray: 
    r"""
    Second order numerical difference along specified dimension

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    Returns
    -------
    ShapeletsArray
        A new instance of an array with the same dimensions as the input array, except 
        in the dimension 'dim', which will have two element less.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[3,7,1],[2,2,9],[5,6,2]])
    >>> sc.diff2(a, 0)
    [1 3 1 1]
        4          9        -15 
    >>> sc.diff2(a, 1)
    [3 1 1 1]
        -10 
          7 
         -5        
    """
    return _pygauss.diff2(a, dim)

def sort(a: ArrayLike, dim: int = 0, asc: bool = True) -> ShapeletsArray: 
    r"""
    Sorts the input array across a dimension

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    asc: bool (default: True)
        Selects ascending (default) or descending order

    Returns
    -------
    ShapeletsArray
        A new instance of an array, with the values sorted across the given dimension.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[3,7,1],[2,2,9],[5,6,2]])
    >>> sc.sort(a)   
    [3 3 1 1]
        2          2          1 
        3          6          2 
        5          7          9     
    """
    return _pygauss.sort(a, dim, asc)

def sort_index(a: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Sorts an array and returns the indices of the original positions across a dimension.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    asc: bool (default: True)
        Selects ascending (default) or descending order

    Returns
    -------
    Tuple
        The first element of the tuple with be a ShapeletsArray with the original indices of the elements
        across the dimension.  The second tuple element will be the sorted array.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[3,7,1],[2,2,9],[5,6,2]])
    >>> positions, sorted = sc.sort_index(a)      
    >>> positions
    [3 3 1 1]
        1          1          0 
        0          2          2 
        2          0          1 
    >>> sorted
    [3 3 1 1]
        2          2          1 
        3          6          2 
        5          7          9     
    """
    return _pygauss.sort_index(a, dim, asc)

def sort_by_key(keys: ArrayLike, data: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""

    Sorts an array across a dimension using an auxiliary array of keys.

    Parameters
    ----------
    keys: ArrayLike
        Input array expression with keys.
    
    a: ArrayLike
        Input array expression

    dim: int (default: 0)
        Axis of the operation. 

    asc: bool (default: True)
        Selects ascending (default) or descending order

    Returns
    -------
    Tuple
        The first element of the result will be a ShapeletsArray with the values sorted; the second element will be 
        the sorted keys.

    Notes
    -----
    It is perfectly valid to have multiple key values repeated in the `keys` parameter.  This method will ensure 
    within a group of same key values, the data will be sorted.        

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4,5,6,7,8])
    >>> k = sc.array([2,4,6,8,2,4,6,8])
    >>> sdata, sk = sc.sort_by_key(k, a, 0, True)
    >>> sdata.T 
    [1 8 1 1]
        1          5          2          6          3          7          4          8 
    >>> sk.T
    [1 8 1 1]
        2          2          4          4          6          6          8          8     
    """
    return _pygauss.sort_by_key(keys, data, dim, asc)

def flatnonzero(a: ArrayLike) -> ShapeletsArray: 
    r"""
    Return indices that are non-zero in the flattened version of a.

    Parameters
    ----------
    a: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A columnar array with indices to non-zero elements. The indices are provided 
        in column-mayor order.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> a = sc.array([[1,2,3], [0,0,0], [4,5,6]])
    >>> sc.flatnonzero(a).T
    [1 6 1 1]
        0          2          3          5          6          8 
    """
    return _pygauss.flatnonzero(a)

def union(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    r"""
    Performs the union of two input arrays

    Parameters
    ----------
    x1: ArrayLike
        First array expression

    x2: ArrayLike
        Second array expression
    
    is_unique: bool (default: False)
        Optimisation flag that indicates the elements in the arrays are 
        already unique and sorted. Setting this flag to True increases the 
        performance of the operation, as it avoids sorting operations.

    Returns
    -------
    ShapeletsArray
        Unique, sorted union of the input arrays.

    Examples
    --------
    >>> import shapelets.compute as sc    
    >>> sc.union([1, 0, -1], [2, -2, 0])
    [5 1 1 1]
        -2 
        -1 
         0 
         1 
         2     
    >>> sc.union([-1, 0, 1], [-2, 0, 2], True)
    [5 1 1 1]
        -2 
        -1 
         0 
         1 
         2     
    """
    return _pygauss.union(x1, x2, is_unique)

def unique(a: ArrayLike, is_sorted: bool = False) -> ShapeletsArray: 
    r"""
    Returns unique values in an array

    Parameters
    ----------
    a: ArrayLike
        Input array expression.

    is_sorted: bool (default: False)
        Optimisation flag to avoid unnecessary sorting operations. 

    Returns
    -------
    ShapeletsArray
        Sorted and unique elements found in the input array.

    Examples
    --------
    >>> import shapelets.compute as sc   
    >>> sc.unique([0, -1, -2, -2,  3, 4, 3]) 
    [5 1 1 1]
        -2 
        -1 
         0 
         3 
         4     
    >>> sc.unique([-2, -2, -1, 0, 3, 3, 4], True)
    [5 1 1 1]
        -2 
        -1 
         0 
         3 
         4     
    """
    return _pygauss.unique(a, is_sorted)

def intersect(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray: 
    r"""
    Finds the intersection between to arrays.

    Parameters
    ----------
    x1: ArrayLike
        First array expression

    x2: ArrayLike
        Second array expression
    
    is_unique: bool (default: False)
        Optimisation flag that indicates the elements in the arrays are 
        already unique and sorted. Setting this flag to True increases the 
        performance of the operation, as it avoids sorting operations.

    Returns
    -------
    ShapeletsArray
        Unique, sorted union of the input arrays.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.intersect([1,2,3], [2,3,4], True)   
    [2 1 1 1]
         2 
         3 
    >>> sc.intersect([3,1,2], [3, 2, 2, 4, -1], False)
    [2 1 1 1]
         2 
         3     
    """
    return _pygauss.intersect(x1, x2, is_unique)

def count_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Counts the non-zero values of an input array according to an array of keys.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the counts.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 1, 0, 1, 1, 0, 0, 1, 0])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, counts = sc.count_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> counts
    [4 1 1 1]
         2 
         2 
         0 
         1 
    """
    return _pygauss.count_by_key(keys, vals, dim)

def max_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the maximum values per key.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the maximums.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, maxvals = sc.max_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> maxvals
    [4 1 1 1]
         2 
         5 
         7 
         9 
    """
    return _pygauss.max_by_key(keys, vals, dim)

def min_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the minimun values per key.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the minimum values.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, minvals = sc.min_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> maxvals
    [4 1 1 1]
         1 
         3 
         6 
         8 
    """
    return _pygauss.min_by_key(keys, vals, dim)

def product_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the product of values per key.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the product values.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, products = sc.product_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> products
    [4 1 1 1]
          2 
         60 
         42 
         72  
    """
    return _pygauss.product_by_key(keys, vals, dim)

def nanproduct_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 1.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the product of values per key, replacing NaN values.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    nan_value: float (default: 1.0)
        Value to replace NaN occurrences.        

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the product values.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np
    >>> v = sc.array([1, 2, 3, np.nan, 5, 6, 7, np.nan, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, products = sc.nanproduct_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> products
    [4 1 1 1]
        2.0000 
       15.0000 
       42.0000 
        9.0000 
    """
    return _pygauss.nanproduct_by_key(keys, vals, nan_value, dim)

def sum_by_key(keys: ArrayLike, vals: ArrayLike, dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the sum of values per key.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the sum of values.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, sums = sc.product_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> sums
    [4 1 1 1]
         3 
        12 
        13 
        17 
    """
    return _pygauss.sum_by_key(keys, vals, dim)

def nansum_by_key(keys: ArrayLike, vals: ArrayLike, nan_value: float = 0.0,  dim: Optional[int] = None) -> Tuple[ShapeletsArray, ShapeletsArray]: 
    r"""
    Finds the sum of values per key, replacing NaN values.

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    nan_value: float (default: 0.0)
        Value to replace NaN occurrences.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    Returns
    -------
    Tuple
        The first element will be a ShapeletsArray with aggregated keys; the second element of the tuple will 
        be the sum of values.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, multiple groups of the same key will appear 
    in the result set.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> import numpy as np
    >>> v = sc.array([1, 2, 3, np.nan, 5, 6, 7, np.nan, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> keys, sums = sc.nansum_by_key(k, v)
    >>> keys
    [4 1 1 1]
         0 
         1 
         0 
         2     
    >>> sums
    [4 1 1 1]
        3.0000 
        8.0000 
       13.0000 
        9.0000 
    """
    return _pygauss.nansum_by_key(keys, vals, nan_value, dim)

def scan_by_key(keys: ArrayLike, vals: ArrayLike, dim: int = 0, op: ScanOp = 'add', inclusive_scan: bool = True) -> ShapeletsArray: 
    r"""
    Generalized scan by key operation 

    Parameters
    ----------
    keys: ArrayLike
        Input 1D array expression with key values. Keys must be 32 bit integers (signed or unsigned); if the method 
        is presented with keys of a different type, they will be silently converted to signed int 32 bits.
    
    vals: ArrayLike
        Input array expression with values.  Keys and Values are expected of the same dimensions.

    dim: Optional int (default: None)
        Dimension for the operation.  If not specified, it will resolve to the first non singleton dimension.

    op: ScanOp (default: 'add')
        Operation to perform.  It could be one of 'add', 'max', 'min' or 'mul'.
    
    inclusive_scan: bool (default: True)
        Indicates if the scan operation is inclusive or exclusive.  In inclusive mode, the first values through 
        an axis will be kept in the result; on the other hand, exclusive scans will introduce Inf, zeros or one 
        values, depending on the operation selected.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the scan operation.  The dimensions of the output would be 
        the same as the input array.

    Notes
    -----
    Keys are not required to be unique or sorted; if that is the case, the operation **will reset** as if the key 
    was processed for the very first time.

    Examples
    --------
    >>> import shapelets.compute as sc  
    >>> v = sc.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> k = sc.array([0, 0, 1, 1, 1, 0, 0, 2, 2], dtype="int32")
    >>> sc.scan_by_key(k, v, op='add')
    [9 1 1 1]
         1 
         3 
         3 
         7 
        12 
         6 
        13 
         8 
        17 
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
        Value utilized to replace all `nan` values.

    inf: float: Defaults to 0.0
        Value utilized to replace `inf` values.

    Returns
    -------
    A new array with no NaN or Inf values.

    Examples
    -------
    >>> import shapelets.compute as sc  
    >>> import numpy as np
    >>> sc.nan_to_num([[np.NaN, 1.0], [np.Inf, 3.0]], 0.0, 2.0)
    [2 2 1 1]
        0.0000     1.0000 
        2.0000     3.0000     
    """
    return _pygauss.nan_to_num(a, nan, inf)

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