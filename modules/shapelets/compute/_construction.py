from __future__ import annotations
from typing import Union, Optional

from .__basic_typing import ArrayLike, Shape, DataTypeLike, ShapeLike, _ScalarLike
from ._array_obj import array as asarray, ShapeletsArray

from . import _pygauss

AnyScalar = _ScalarLike

def linspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, axis: int = 0, dtype: DataTypeLike = "float32") -> ShapeletsArray: 
    """
    Creates a new array whose values are evenly spaced over an interval.

    Parameters
    ----------
    start: ArrayLike
        Inclusive value where the series start.  This value could be either a scalar value or a tensor.

    end: ArrayLike
        End value for the series.  This parameter could be either a scalar value or a tensor.  When both 
        ``start`` and ``end`` parameters are tensors, both have to denote the same shape.

    num: int, defaults to 50
        Number of steps between ``start`` and ``end``.

    endpoint: bool, defaults to True
        Should the ``end`` value be included in the final series. 

    axis: int, defaults to 0
        Determines the axis where the sequence is generated.ß

    dtype: DataTypeLike, defaults to 'float32'
        Tensor data type.

    Returns
    -------
    ShapeletsArray
        A new array instance 

    See Also
    --------
    logspace
        Similar functionality of a logarithmic scale
    geomspace
        Similar behaviour over a geometric scale
    arange
    iota


    Examples
    --------
    Create a column vector starting from 0 to 1 with three elements:

    >>> import shapelets.compute as sc
    >>> sc.linspace(0, 1, 3)
    [3 1 1 1]
        0.0000 
        0.5000 
        1.0000 
    
    Setting up the axis:

    >>> sc.linspace(0, 1, 3, axis=2)
    [1 1 3 1]
        0.0000 
        0.5000 

    Using arrays as parameters:

    >>> sc.linspace([-10, 0], [10, 10], 5)
    [5 2 1 1]
        -10          0 
        - 5          2 
          0          4 
          5          6 
         10          8     

    Mixed specifications:

    >>> sc.linspace([[-1,2],[3,-4]], 5, 5)
    [5 2 2 1]
        -1.0000     3.0000 
         0.5000     3.5000 
         2.0000     4.0000 
         3.5000     4.5000 
         5.0000     5.0000 

         2.0000    -4.0000 
         2.7500    -1.7500 
         3.5000     0.5000 
         4.2500     2.7500 
         5.0000     5.0000     

    """
    return _pygauss.linspace(start, end, num, endpoint, axis, dtype)

def logspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, axis: int = 0, base: float = 10.0, dtype: DataTypeLike = "float32") -> ShapeletsArray: 
    """
    Creates a new array whose values are evenly spaced over a log scale.

    In linear space, the sequence starts at ``base ** start`` (base to the power of start) and 
    ends with ``base ** stop``.

    Parameters
    ----------
    start: ArrayLike
        ``base ** start`` (inclusive) will be the value where the series start.  This 
        parameter could be either a scalar value or a tensor.

    end: ArrayLike
        ``base ** start`` will be the end value of the series.  This parameter could be either 
        a scalar value or a tensor.  When both ``start`` and ``end`` parameters are tensors, 
        both have to denote the same shape.

    num: int, defaults to 50
        Number of steps between ``start`` and ``end``.

    endpoint: bool, defaults to True
        Should the ``end`` value be included in the final series. 

    axis: int, defaults to 0
        Determines the axis where the sequence is generated.

    base: float, defaults to 10
        Base of the series

    dtype: DataTypeLike, defaults to 'float32'
        Tensor data type.

    Returns
    -------
    ShapeletsArray
        A new array instance 

    Examples
    --------
    This example highlights the usage of ``endpoint``:

    >>> sc.logspace(2.0, 5.0, num=3, endpoint=False, base=2.0)
        [3 1 1 1]
           4.0000 
           8.0000 
          16.0000 
    >>> sc.logspace(2.0, 5.0, num=4, endpoint=True, base=2.0)
        [4 1 1 1]
           4.0000 
           8.0000 
          16.0000 
          32.0000     
    """
    return _pygauss.logspace(start, end, num, endpoint, axis, base, dtype)

def geomspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, axis: int = 0, dtype: DataTypeLike = "float32") -> ShapeletsArray: 
    """
    Creates a new array whose values are evenly spaced over a geometric scale.

    This function is quite simmilar to :obj:`~shapelelets.compute.logspace`, but with the 
    start and end values explicitely stated.

    Parameters
    ----------
    start: ArrayLike
        Inclusive value where the series start.  This value could be either a scalar value or a tensor.

    end: ArrayLike
        End value for the series.  This parameter could be either a scalar value or a tensor.  When both 
        ``start`` and ``end`` parameters are tensors, both have to denote the same shape.

    num: int, defaults to 50
        Number of steps between ``start`` and ``end``.

    endpoint: bool, defaults to True
        Should the ``end`` value be included in the final series. 

    axis: int, defaults to 0
        Determines the axis where the sequence is generated.ß

    dtype: DataTypeLike, defaults to 'float32'
        Tensor data type.

    Returns
    -------
    ShapeletsArray
        A new array instance 
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.geomspace(1, 256, num=9, axis = 1)
    [1 9 1 1]
        1.00    2.00    4.00    8.00    16.00    32.00    64.00   128.00   256.00 

    """
    return _pygauss.geomspace(start, end, num, endpoint, axis, dtype)

def arange(start: Union[int, float], stop: Union[int, float], step: Union[int, float], dtype: DataTypeLike = "float32") -> ShapeletsArray:
    """
    Creates a new array whose values are evenly spaced in a given interval.
    """
    return _pygauss.arange(start, stop, step, dtype)

def empty(shape: ShapeLike, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Constructs an empty tensor.

    Use this construct to reserve a memory zone of the current device in preparation for 
    holding some future results.  

    The memory is not guaranteed to be initialized; use `full`, `ones`, `zeros` or other 
    initialization methods if you need the memory to be fully initialized before acceesing it.

    Parameters
    ----------
    shape : ShapeLike
        A scalar or a sequence of integers defining the number of elements per tensor dimension.

    dtype: DataTypeLike
        The type of elements this tensor is going to hold

    Returns
    -------   
    ShapeletsArray
        An uninitialized tensor where all elements are of type `dtype`.

    See Also
    --------
    full
    ones
    zeros

    Examples
    --------

    >>> import shapelets.compute as sc
    >>> sc.empty((3,3))
        [3 3 1 1]
            nan        nan        nan 
            nan        nan        nan 
            nan        nan        nan          
    """
    return _pygauss.empty(shape, dtype)

def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N: int
        Number of rows in the output.

    M: int, optional
        Number of columns in the output. If None, defaults to `N`.

    k: int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.

    dtype: data-type, optional
        Data-type of the returned array.

    Returns
    -------
    ShapeletsArray
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.    
    """
    return _pygauss.eye(N, M, k, dtype)

def full(shape: ShapeLike, fill_value: AnyScalar, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        
    fill_value : scalar or array_like
        Fill value.

    dtype : data-type, optional
        The desired data-type for the array  If not specified it will
        default to "float".
    """   
    return _pygauss.full(shape, fill_value, dtype)

def full_like(a: ArrayLike, fill_value: AnyScalar, shape: Optional[ShapeLike] = None, dtype: Optional[DataTypeLike] = None) -> ShapeletsArray:
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a: ArrayLike
        Template array to extract shape and type.
    
    fill_value: AnyScalar
        Constant to fill the array
    
    shape: ShapeLike, optional (default: None)
        Overrides the shape implied by the template array
    
    dtype: DataTypeLike, optional (default: None)
        Overrides the type implied by the template array

    Returns
    -------
    ShapeletsArray
        A new array instance fill with a constant value

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2], [3,4]])
    >>> sc.full_like(a, 3)
    [2 2 1 1]
            3          3 
            3          3 
    >>> sc.full_like(a, 3, dtype="float64")
    [2 2 1 1]
        3.0000     3.0000 
        3.0000     3.0000 
    """
    template = asarray(a)
    t_shape = shape if shape is not None else template.shape
    t_dtype = dtype if dtype is not None else template.dtype
    return _pygauss.full(t_shape, fill_value, t_dtype)

def identity(shape: Shape, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an identity tensor with diagonal values set to one.

    Parameters
    ----------
    shape : Shape
        When set to an integer, n, it will return return an identity matrix
        of nxn.  Otherwise, use a tuple to specify the exact dimensions for
        the tensor where all the diagonal elements will be set to 1.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : Array
        Array of shape ``shape`` whose diagonal elements are all set to zero.
    """        
    return _pygauss.identity(shape, dtype)

def zeros(shape: ShapeLike, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with all its elements set to zero.
    """
    return _pygauss.zeros(shape, dtype)

def zeros_like(a: ArrayLike, shape: Optional[ShapeLike] = None, dtype: Optional[DataTypeLike] = None) -> ShapeletsArray:
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a: ArrayLike
        Template array to extract shape and type.
    
    shape: ShapeLike, optional (default: None)
        Overrides the shape implied by the template array
    
    dtype: DataTypeLike, optional (default: None)
        Overrides the type implied by the template array

    Returns
    -------
    ShapeletsArray
        A new array instance fill with zeros

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2], [3,4]])
    >>> sc.zeros_like(a)
    [2 2 1 1]
            0          0 
            0          0 
    >>> sc.zeros_like(a, dtype="float64")
    [2 2 1 1]
        0.0000     0.0000 
        0.0000     0.0000 

    """
    template = asarray(a) 
    t_shape = shape if shape is not None else template.shape
    t_dtype = dtype if dtype is not None else template.dtype
    return _pygauss.zeros(t_shape, t_dtype)    

def ones(shape: ShapeLike, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with all its elements set to one.

    """
    return _pygauss.ones(shape, dtype)

def ones_like(a: ArrayLike, shape: Optional[ShapeLike] = None, dtype: Optional[DataTypeLike] = None) -> ShapeletsArray:
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a: ArrayLike
        Template array to extract shape and type.
    
    shape: ShapeLike, optional (default: None)
        Overrides the shape implied by the template array
    
    dtype: DataTypeLike, optional (default: None)
        Overrides the type implied by the template array

    Returns
    -------
    ShapeletsArray
        A new array instance fill with ones

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2], [3,4]])
    >>> sc.ones_like(a)
    [2 2 1 1]
            1          1 
            1          1 
    >>> sc.zeros_like(a, dtype="float64")
    [2 2 1 1]
        1.0000     1.0000 
        1.0000     1.0000  

    """
    template = asarray(a) 
    t_shape = shape if shape is not None else template.shape
    t_dtype = dtype if dtype is not None else template.dtype
    return _pygauss.ones(t_shape, t_dtype) 

def diag(a: ArrayLike, index: int = 0, extract: bool = False) -> ShapeletsArray:
    """
    Operates with diagonals
    
    Parameters
    ----------
    a: ArrayLike
        Input array

    index: int, defaults to 0
        Relative diagonal index, where 0 corresponds to the main diagonal.
    
    extract: bool, defaults to False
        Using extract parameter one is able to either create a diagonal matrix 
        from a vector (false) or extract a diagonal from a matrix to a vector (true)

    Returns
    -------
    ShapeletsArray
        Either a columnar vector when extract is True or a diagonal matrix where all 
        elements are set to zero besides the diagonal ``index``.
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3])
    >>> sc.diag(a, 0)
    [3 3 1 1]
            1          0          0 
            0          2          0 
            0          0          3    
    >>> sc.diag(a, 1)
    [4 4 1 1]
            0          1          0          0 
            0          0          2          0 
            0          0          0          3 
            0          0          0          0     
    >>> b = sc.iota((3,3))
    [3 3 1 1]
        0.0000     3.0000     6.0000 
        1.0000     4.0000     7.0000 
        2.0000     5.0000     8.0000 
    >>> sc.diag(b, extract=True)
    [3 1 1 1]
        0.0000 
        4.0000 
        8.0000  
    >>> sc.diag(b, -1, extract=True)
    [2 1 1 1]
        1.0000 
        5.0000 

    """
    return _pygauss.diag(a, index, extract)

def iota(shape: ShapeLike, tile: Shape = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Create an sequence ``[0, shape.elements() - 1]`` and modify to specified dimensions 
    dims and then tile it according to tile

    Parameters
    ----------
    shape: ShapeLike
        Determines the initial shape of the array.

    tile: ShapeLike
        Condenses the tiling in one single parameter.  For example: ``(2,3)`` implies repeat twice 
        in the row axis direction and three times in the column direction.
    
    dtype: DataTypeLike
        Determines the type of array to be created.

    Returns
    -------
    ShapeletsArray
        A new instance on an array
    
    Examples
    --------
    Generate a 5 by 3 matrix, numerating each entry in column major order and then tile once along 
    dimension 0 and twice on dimension 1:

    >>> import shapelets.compute as sc
    >>> sc.iota((5,3), (1,2), "int32")
    [5 6 1 1]
        0   5   10   0   5   10 
        1   6   11   1   6   11 
        2   7   12   2   7   12 
        3   8   13   3   8   13 
        4   9   14   4   9   14 
    """
    return _pygauss.iota(shape, tile, dtype)

def range(shape: ShapeLike, seq_dim: int = 0, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with [0, n] values along the seq_dim which is tiled across other dimensions.

    Parameters
    ----------
    shape: ShapeLike
        Dimensions of the array

    seq_dim: int, defaults to 0
        Dimension that generates the sequence.  

    dtype: DataTypeLike
        Final type of the generated array.
    
    Returns
    -------
    ShapeletsArray
        A new array instance.
    
    Examples
    --------
    >>> sc.range((3,4), seq_dim=0)
    [3 4 1 1]
        0.0000     0.0000     0.0000     0.0000 
        1.0000     1.0000     1.0000     1.0000 
        2.0000     2.0000     2.0000     2.0000 

    >>> sc.range((3,4), seq_dim=1)
    [3 4 1 1]
        0.0000     1.0000     2.0000     3.0000 
        0.0000     1.0000     2.0000     3.0000 
        0.0000     1.0000     2.0000     3.0000 

    """
    return _pygauss.range(shape, seq_dim, dtype)

__all__ = [
    "geomspace", "logspace", "linspace", "arange", "arange", "empty", "eye", 
    "identity", "full", "zeros", "ones", "diag", "iota", "range",
    "zeros_like", "ones_like", "full_like"
]
