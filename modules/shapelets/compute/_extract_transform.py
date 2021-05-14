from __future__ import annotations
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .__basic_typing import ArrayLike, Shape, DataTypeLike
from ._array_obj import ShapeletsArray
from . import _pygauss

BorderType = Literal['clampedge', 'periodic', 'symmetric', 'zero']


def __pygauss_fill_type(type: BorderType):
    if type == 'clampedge':
        return _pygauss.BorderType.ClampEdge
    elif type == 'periodic':
        return _pygauss.BorderType.Periodic
    elif type == 'symmetric':
        return _pygauss.BorderType.Symmetric
    elif type == 'zero':
        return _pygauss.BorderType.Zero
    else:
        raise ValueError("Unknown border type")


def cast(array_like: ArrayLike, dtype: DataTypeLike) -> ShapeletsArray:
    """
    Changes the type of elements in an array

    Parameters
    ----------
    array_like: ArrayLike
        The input array
    
    dtype: DataTypeLike
        Expression evaluating to a data type
    
    Returns
    -------
    ShapeletsArray
        A new array whose elements are of the specified type.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.cast([1,2,3,4], np.complex64)
    [4 1 1 1]
        (1.0000,0.0000) 
        (2.0000,0.0000) 
        (3.0000,0.0000) 
        (4.0000,0.0000) 
    >>> a = np.array([1,2,3,4])
    >>> sc.cast(a, "float32")
    [4 1 1 1]
        1.0000 
        2.0000 
        3.0000 
        4.0000    
    """
    return _pygauss.cast(array_like, dtype)


def flat(array_like: ArrayLike) -> ShapeletsArray:
    """
    Flattens the dimensions of the array into a columnar vector.

    Please note the flatten process will follow a column major order.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A columnar vector whose elements are the same ones as the 
        original one.  
    
    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.flat([[1,2],[3,4]])
    [4 1 1 1]
            1 
            3 
            2 
            4     

    """
    return _pygauss.flat(array_like)


def flip(array_like: ArrayLike, dimension: int = 0) -> ShapeletsArray:
    """
    Flips the elements of an array in a particular dimension

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    dimension: int (defaults: 0)
        Dimension to flip
    
    Returns
    -------
    ShapeletsArray
        New array instance where all the elements in a particular dimension
        have been flipped.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> a.display()
    [3 3 1 1]
         1          2          3 
         4          5          6 
         7          8          9 
    >>> sc.flip(a, 0)
    [3 3 1 1]
            7          8          9 
            4          5          6 
            1          2          3     
    >>> sc.flip(a, 1)
    [3 3 1 1]
            3          2          1 
            6          5          4 
            9          8          7     
    """
    return _pygauss.flip(array_like, dimension)


def join(lst: List[ArrayLike], dimension: int = 0) -> ShapeletsArray:
    """
    Joins arrays on a particular dimension.

    Parameters
    ----------
    lst: List
        List of ArrayLike expressions, which must resolve to the same underlying type.

    dimension: int (defaults: 0)
        Dimension for the join operation

    Returns
    -------
    ShapeletsArray
        A new array instance with all the individual input arrays joined on the 
        prescribed dimension.
    
    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> import numpy as np
    >>> a = [1,2]
    >>> b = np.array([3,4])
    >>> c = sc.array([5,6])
    >>> sc.join([a,b,c], 0)
    [6 1 1 1]
            1 
            2 
            3 
            4 
            5 
            6 
    >>> sc.join([a,b,c],1)
    [2 3 1 1]
            1          3          5 
            2          4          6     
    >>> sc.join([a,b,c], 3)
    [2 1 3 1]
            1 
            2 

            3 
            4 

            5 
            6                 
    """
    return _pygauss.join(lst, dimension)


def lower(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray:
    """
    Returns the lower triangular matrix of an input array

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    unit_diag: bool (defaults: False)
        When set to False, or unset, the elements of the main diagonal matrix 
        will be returned.  Alternatively, set it to True to return ``1s`` in 
        the main diagonal positions.
    
    Returns
    -------
    ShapeletsArray
        A new array instance with all original values from the lower triangular 
        part of the input matrix.  All the other elements will be set to 0.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> sc.lower(a)
    [3 3 1 1]
        1          0          0 
        4          5          0 
        7          8          9     
    >>> sc.lower(a, True)
    [3 3 1 1]
        1          0          0 
        4          1          0 
        7          8          1 
    """
    return _pygauss.lower(array_like, unit_diag)


def pad(array_like: ArrayLike, begin: Shape, end: Shape, fill_type: BorderType) -> ShapeletsArray:
    """
    Pads an array

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    begin: Shape
        Full 4 dimensional tuple specifying how many elements to add at the beggining of the 
        input array.  Negative values are not permitted; 0 implies no changes on a particular 
        dimension.
    
    end: Shape
        Full 4 dimensional tuple, representing the padding at the end of the array. Negative 
        values are not permitted; 0 implies no changes on a particular 
        dimension.
    
    fill_type: BorderType {'clampedge', 'periodic', 'symmetric', 'zero'}
        Determines the values for the new padded elements. 

    Returns
    -------
    ShapeletsArray
        Padded version of the original array.
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2],[3,4]])

    No pad at the begining but one extra row at the end, initialized with 'zeros':

    >>> sc.pad(a, (0,0,0,0), (1,0,0,0), 'zero')
    [3 2 1 1]
        1          2 
        3          4 
        0          0 

    Adding two extra rows and two extra column at both ends, using 'symmetric':

    >>> sc.pad(a, (1,1,0,0), (1,1,0,0), 'symmetric')
    [4 4 1 1]
            1          1          2          2 
            1          1          2          2 
            3          3          4          4 
            3          3          4          4 

    Same as before, but using 'periodic':

    >>> sc.pad(a, (1,1,0,0), (1,1,0,0), 'periodic')
    [4 4 1 1]
            4          3          4          3 
            2          1          2          1 
            4          3          4          3 
            2          1          2          1 


    Finally, using 'clampedge':

    >>> sc.pad(a, (1,1,0,0), (1,1,0,0), 'clampedge')
    [4 4 1 1]
            1          1          2          2 
            1          1          2          2 
            3          3          4          4 
            3          3          4          4     
    """
    return _pygauss.pad(array_like, begin, end, __pygauss_fill_type(fill_type))


def reorder(array_like: ArrayLike, x: int, y: int = 1, z: int = 2, w: int = 3) -> ShapeletsArray:
    """
    Reorganises the information in the underlying array by exchanging dimensions

    Dimension indices start at zero (rows or 1st dimension) and finishes at three (4th dimension)

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    x: int
        Index of the dimension that will be now the 1st dimension
    y: int (defaults: 1, implying no changes)
        Index of the dimension that will be now the 2st dimension
    z: int (defaults: 2, implying no changes)
        Index of the dimension that will be now the 3st dimension
    w: int (defaults: 3, implying no changes)
        Index of the dimension that will be now the 4st dimension

    Returns
    -------
    ShapeletsArray
        New array instance with the new organisation.

    Notes
    -----
    This method is really usefull when dealing with operations that have 
    built in vectorization, like :obj:`~shapelets.compute.convolve1`, as 
    it allows to reorganise arrays before triggering parallel computations.

    Examples
    --------
    Given nxM signals and mxP filters, convolve all of them in parallel in one
    single device operation:

    >>> import shapelets.compute as sc 
    >>> M = 100; n = 50
    >>> P = 10; m = 5
    >>> signals = sc.random.randn((n, M))
    >>> filters = sc.random.randn((m, P))
    >>> print(signals.shape, filters.shape)
    (50, 100) (5, 10)
    >>> filters = sc.reorder(filters, 0, 2, 1, 3) # swap 1 and 2
    >>> filters.shape
    (5, 1, 10)
    >>> r = sc.convolve1(signals, filters, 'expand')
    >>> r.shape
    (54, 100, 10)   # 54 Due to 'expand', 100 signals, 10 filters
    """
    return _pygauss.reorder(array_like, x, y, z, w)


def reshape(array_like: ArrayLike, shape: Shape) -> ShapeletsArray:
    """
    Changes the dimensions of the array without modifying the underlying data

    Please note that the element counts implied by the new dimensionality should 
    be the same as the original one.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression.
    
    shape: Shape
        A shape expression, `int` or `tuple of ints` defining the new shape of the array.

    Returns
    -------
    ShapeletsArray
        A new instance of an array with the new dimensionality.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4])
    >>> b = sc.reshape(a, (2,2)) 
    >>> b.display()
    [2 2 1 1]
        1          3 
        2          4     
    >>> sc.reshape(a, (1,4))
    [1 4 1 1]
        1          2          3          4     
    """
    return _pygauss.reshape(array_like, shape)


def shift(array_like: ArrayLike, x: int, y: int = 0, z: int = 0, w: int = 0) -> ShapeletsArray:
    """
    Shifts or rotates an array

    Positive shifts will rotate to the "right", whilst negative shifts will rotate to the "left". 
    Shift quatities will be subject to the modulus of the dimension cardinality.

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    x:int 
        Elements to swift downwards (positive) or upwards the 1st dimension (rows).

    y:int (default: 0)
        Elements to swift right (positive) or left the 2st dimension (columns).

    z:int (default: 0)
        Elements to swift on the 3rd dimension 

    w:int (default: 0)
        Elements to swift on the 4th dimension 

    Returns
    -------
    ShapeletsArray
        New array instance with the effects of the rotation.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> sc.shift(a, -1) # rows upwards by one
    [3 3 1 1]
        4          5          6 
        7          8          9 
        1          2          3 
    >>> sc.shift(a, 0, 2) # columns to right by 2
    [3 3 1 1]
        2          3          1 
        5          6          4 
        8          9          7 
    >>> sc.shift(a, -1, 2) # both effects applied simultaneously.
    [3 3 1 1]
        5          6          4 
        8          9          7 
        2          3          1     
    """
    return _pygauss.shift(array_like, x, y, z, w)


def tile(array_like: ArrayLike, x: int, y: int = 1, z: int = 1, w: int = 1) -> ShapeletsArray:
    """
    Tiles the input array, duplicating contents across dimensions.

    This operation doesn't imply the allocation of new memory on the device, but helps to 
    shape accurately the dimensionality of the operation in those occasions where broadcasting 
    rules are not appropriate.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    x: int
        Number of times to tile the first dimension.  One implies no operation.

    y: int (default: 1)
        Number of times to tile the second dimension

    z: int (default: 1)
        Number of times to tile the third dimension

    w: int (default: 1)
        Number of times to tile the fourth dimension
    
    Returns
    -------
    ShapeletsArray
        New array instance with the outcome of the tiling operation.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.tile([[1,2],[3,4]], 1, 2) # x2 columns
    [2 4 1 1]
        1          2          1          2 
        3          4          3          4     
    >>> sc.tile([[1,2],[3,4]], 2, 2) # x2 columns x2 rows
    [4 4 1 1]
        1          2          1          2 
        3          4          3          4 
        1          2          1          2 
        3          4          3          4    
    >>> sc.tile([[1,2],[3,4]], 1, 1, 1, 2) # tile over the fourth dimension
    [2 2 1 2]
         1          2 
         3          4 


         1          2 
         3          4     
    """
    return _pygauss.tile(array_like, x, y, z, w)

def transpose(array_like: ArrayLike, conjugate: bool = False) -> ShapeletsArray:
    """
    Transposes the input array

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression

    conjugate: bool (default: False)
        When set to true, a transpose conjugate operation will occur.

    Returns
    -------
    ShapeletsArray
        A transpose, or transpose conjugate, of the original array.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.transpose([1+1j, 2-2j])
    [1 2 1 1]
        (1.0000,1.0000)          (2.0000,-2.0000) 
    >>> sc.transpose([1+1j, 2-2j], True)
    [1 2 1 1]
        (1.0000,-1.0000)          (2.0000,2.0000)    
    """
    return _pygauss.transpose(array_like, conjugate)


def upper(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray:
    """
    Returns the upper triangular matrix of an input array

    Parameters
    ----------
    array_like: ArrayLike
        Input array
    
    unit_diag: bool (defaults: False)
        When set to False, or unset, the elements of the main diagonal matrix 
        will be returned.  Alternatively, set it to True to return ``1s`` in 
        the main diagonal positions.
    
    Returns
    -------
    ShapeletsArray
        A new array instance with all original values from the upper triangular 
        part of the input matrix.  All the other elements will be set to 0.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> sc.upper(a)
    [3 3 1 1]
        1          2          3 
        0          5          6 
        0          0          9     
    >>> sc.upper(a, True)
    [3 3 1 1]
        1          2          3 
        0          1          6 
        0          0          1  
    """
    return _pygauss.upper(array_like, unit_diag)


def where(condition: ArrayLike, x: ArrayLike = None, y: ArrayLike = None) -> ShapeletsArray:
    """
    Selectively chooses elements from two input arrays based on a condition (element-wise).

    All input arrays must be of the same dimensions; use :obj:`~shapelets.compute.tile` to 
    adjust the dimensions as no broadcasting rules are applied in this method.

    Parameters
    ----------
    condition: ArrayLike
        Input boolean array.
    
    x: ArrayLike
        Input array whose elements :math:`x_{indx}` would be selected if :math:`c_{indx} = True`.

    y: ArrayLike
        Input array whose elements :math:`y_{indx}` would be selected if :math:`c_{indx} = False`.

    Notes
    -----
    If ``x`` and ``y`` arrays are of the same type, the result will be of type ``y``.

    Returns
    -------
    ShapeletsArray
        New array instance based on element wise evaluation of the condition array.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = [1, 1]
    >>> b = [2, 2]
    >>> c = [True, False]
    >>> sc.where(c, a, b)
    [2 1 1 1]
        1 
        2     
    """
    return _pygauss.where(condition, x, y)


def unpack(a: ArrayLike, wx: int, wy: int, sx: int, sy: int, px: int = 0, py: int = 0, is_column: bool = True):
    """
    Rearranges window sections of an input into columns or rows.

    For inputs that have more than two dimensions, the unpack operation will be applied to each 2D slice of the input.

    For a thorough explanation of this method, consult the `ArrayFire documentation <https://arrayfire.org/docs/group__image__func__unwrap.htm>`_

    Parameters
    ----------
    a: ArrayLike
        Input array

    wx: int    
        Window size along 1st dimension 
    wy: int    
        Window size along 2st dimension 
    sx: int    
        Stride along 1st dimension 
    sy: int    
        Stride along 2st dimension 
    px: int (default: 0)       
        Padding along 1st dimension 
    py: int (default: 0)      
        Padding along 2st dimension
    is_column: bool (default: True)       
        Determines if the section becomes a column (True) or a row (False).

    Returns
    -------
    ShapeletsArray
        A new array instance.
    
    Examples
    --------
    Given
    
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4,5,6,7])

    Create window batches of 3 elements
    
    >>> sc.unpack(a, 3, 1, 3, 1)
    [3 2 1 1]
        1          4 
        2          5 
        3          6 
    
    Create moving windows of 3 elements:

    >>> sc.unpack(a, 3, 1, 1, 1)
    [3 5 1 1]
        1          2          3          4          5 
        2          3          4          5          6 
        3          4          5          6          7 

    """
    return _pygauss.unpack(a, wx, wy, sx, sy, px, py, is_column)


def pack(a: ArrayLike, ox: int, oy: int, wx: int, wy: int, sx: int, sy: int, px: int = 0, py: int = 0, is_column: bool = True):
    """
    Reverses the :obj:`~shapelets.compute.unpack` operation

    For a thorough explanation of this method, consult the `ArrayFire documentation <https://arrayfire.org/docs/group__image__func__wrap.htm>`_

    Parameters
    ----------
    a: ArrayLike
        Input array

    ox: int    
        Output size for 1st dimension
    oy: int    
        Output size for 2st dimension
    wx: int    
        Window size along 1st dimension 
    wy: int    
        Window size along 2st dimension 
    sx: int    
        Strade along 1st dimension 
    sy: int    
        Strade along 2st dimension
    px: int (default: 0)    
        Padding along 1st dimension 
    py: int (default: 0)       
        Padding along 2st dimension
    is_column: bool (default: True)       
        Determines if an output patch is formed from a column (if true) or a row (if false)

    Returns
    -------
    ShapeletsArray
        A new array instance

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4,5,6,7,8,9])
    >>> w = sc.unpack(a, 3, 1, 3, 1)
    >>> sc.pack(w, 9, 1, 3, 1, 3, 1)
    [9 1 1 1]
        1 
        2 
        3 
        4 
        5 
        6 
        7 
        8 
        9     
    """
    return _pygauss.pack(a, ox, oy, wx, wy, sx, sy, px, py, is_column)


__all__ = [
    "pad", "lower", "upper", "reshape", "flat", "flip", "reorder",
    "shift", "tile", "transpose", "cast", "join", "where",
    "pack", "unpack",
    "BorderType"
]
