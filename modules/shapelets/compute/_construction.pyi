from typing import overload, Union, Optional
from .__basic_typing import ArrayLike, Shape, DataTypeLike, ShapeLike, _ScalarLike
from ._array_obj import ShapeletsArray

AnyScalar = _ScalarLike

def linspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, dtype: DataTypeLike = "float32") -> ShapeletsArray: ...

def logspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, axis: float = 10.0, dtype: DataTypeLike = "float32") -> ShapeletsArray: ...

def geomspace(start: ArrayLike, end: ArrayLike, num: int = 50, endpoint: bool = True, dtype: DataTypeLike = "float32") -> ShapeletsArray: ...

@overload
def arange(stop: Union[int, float], dtype: DataTypeLike = "float32") -> ShapeletsArray: ...

@overload
def arange(start: Union[int, float], stop: Union[int, float], step: Union[int, float], dtype: DataTypeLike = "float32") -> ShapeletsArray: ...

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
    I : tensor of shape (N,M)
        An uninitialized tensor where all elements are of type `dtype`.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.empty((3,3))
        [3 3 1 1]
            nan        nan        nan 
            nan        nan        nan 
            nan        nan        nan          
    """

def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.

    Returns
    -------
    I : tensor of shape (N,M)
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.    
    """

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

def zeros(shape: ShapeLike, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with the given dimensions with all its elements set to zero.
    """

def ones(shape: ShapeLike, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with the given dimensions with all its elements set to one.
    """

def diag(a: ShapeletsArray, index: int = 0, extract: bool = False) -> ShapeletsArray:
    """
    Operates with diagonals
    
    Using extract parameter one is able to either create a diagonal matrix from a vector (false) or 
    extract a diagonal from a matrix to a vector (true)

    """

def iota(shape: ShapeLike, tile: Shape = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Create an sequence [0, shape.elements() - 1] and modify to specified dimensions dims and then tile it according to tile
    """

def range(shape: Shape, seq_dim: int = -1, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
    """
    Creates an array with [0, n] values along the seq_dim which is tiled across other dimensions.
    """
