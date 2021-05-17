from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
from . import _pygauss

def iscomplex(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Signals all those positions with imaginary values distinct to zero

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new boolean array matching the dimensions of the input array with 
        True values in those positions where the input array have imaginary 
        values distinct to zero.
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1, 2j, 3+0j])
    >>> sc.iscomplex(a)
    [3 1 1 1]
        0 
        1 
        0 
    """
    return _pygauss.iscomplex(array_like)

def isfinite(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Signals all those elements not infinite or not NaN.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new boolean array matching the dimensions of the input array with 
        True values in those positions where the input array has no NaN or 
        Inf values
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([1, np.Inf, np.nan])
    >>> sc.isfinite(a)
    [3 1 1 1]
        1 
        0 
        0 
    """
    return _pygauss.isfinite(array_like)

def isinf(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Signals all those elements set to positive or negative infinity.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new boolean array matching the dimensions of the input array with 
        True values in those positions where the input array has Inf values
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([1, np.Inf, np.nan])
    >>> sc.isinf(a)   
    [3 1 1 1]
        0 
        1 
        0      
    """
    return _pygauss.isinf(array_like)

def isnan(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Signals all those elements set to NaN

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new boolean array matching the dimensions of the input array with 
        True values in those positions where the input array has NaN values
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([1, np.Inf, np.nan])
    >>> sc.isnan(a)   
    [3 1 1 1]
        0 
        0
        1     
    """
    return _pygauss.isnan(array_like)

def isreal(array_like: ArrayLike) -> ShapeletsArray: 
    """
    Signals all those elements whose value is real.

    Parameters
    ----------
    array_like: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new boolean array matching the dimensions of the input array with 
        True values in those positions where the input array has real values.
    
    Notes
    -----
    Infinite and NaN values will be also identified by this test.
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> import numpy as np
    >>> a = sc.array([1, np.Inf, np.nan, 2+0j, 3j])
    >>> sc.isreal(a)   
    [3 1 1 1]
         1 
         1 
         1 
         1 
         0     
    """
    return _pygauss.isreal(array_like)

def logical_and(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Performs an element-wise logical ``and`` operation

    Inputs are not required to be boolean arrays; ``0s`` will be interpreted 
    as False and any other value will resolve to True.

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,0,0,1])
    >>> b = a.T
    >>> sc.logical_and(a,b)
    [4 4 1 1]
        1          0          0          1 
        0          0          0          0 
        0          0          0          0 
        1          0          0          1     
    """
    return _pygauss.logical_and(left, right)

def logical_not(array_like: ArrayLike) -> ShapeletsArray: 
    """

    """
    return _pygauss.logical_not(array_like)

def logical_or(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Performs an element-wise logical ``or`` operation

    Inputs are not required to be boolean arrays; ``0s`` will be interpreted 
    as False and any other value will resolve to True.

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([True,False,True,False])
    >>> b = a.T
    >>> sc.logical_or(a,b)
    [4 4 1 1]
        1          1          1          1 
        1          0          1          0 
        1          1          1          1 
        1          0          1          0       
    """
    return _pygauss.logical_or(left, right)

def not_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise non equality test between two arrays

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.not_equal(a, 1)
    [4 1 1 1]
        0 
        1 
        1 
        0     
    """
    return _pygauss.not_equal(left, right)

def equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise equality test between two arrays

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.equal(a, 1)
    [4 1 1 1]
        1 
        0 
        0 
        1     
    """
    return _pygauss.equal(left, right)

def greater(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise test for ``left > right``

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.greater(a, 2)
    [4 1 1 1]
        0 
        1 
        1 
        0        
    """
    return _pygauss.greater(left, right)

def greater_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise test for ``left >= right``

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.greater_equal(a, 3)
    [4 1 1 1]
        0 
        1 
        1 
        0        
    """
    return _pygauss.greater_equal(left, right)

def less(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise test for ``left < right``

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.less(a, 2)
    [4 1 1 1]
        1
        0 
        0 
        1       
    """
    return _pygauss.less(left, right)

def less_equal(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Element-wise test for ``left <= right``

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new boolean array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,3,3,1])
    >>> sc.less_equal(a, 1)
    [4 1 1 1]
        1
        0 
        0 
        1       
    """
    return _pygauss.less_equal(left, right)


__all__ = [
    "isfinite", "isinf", "isnan", "iscomplex", "isreal", "logical_and", "logical_or", 
    "logical_not", "equal", "not_equal", "greater", "greater_equal", "less", "less_equal"
]