from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

def add(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Adds arguments element-wise

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation.

    Notes
    -----
    This operation is equivalent to the standard '+' operators between arrays.  

    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,0,0,1]).T
    >>> sc.add(a, 1)
    [1 4 1 1]
        2          1          1          2 
    """
    return _pygauss.add(left, right)

def bitwise_and(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Bitwise ``and`` operation applied element-wise

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([0,1,2,3]).T
    >>> sc.bitwise_and(a, 1)
    [1 4 1 1]
        0          1          0          1     
    """
    return _pygauss.bitwise_and(left, right)

def bitwise_or(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Bitwise ``or`` operation applied element-wise

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([0,1,2,3]).T
    >>> sc.bitwise_or(a, 1)
    [1 4 1 1]
        1          1          3          3      
    """
    return _pygauss.bitwise_or(left, right)
    
def bitwise_xor(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Bitwise ``xor`` operation applied element-wise

    Parameters
    ----------
    left: ArrayLike 
        Input array expression    

    right: ArrayLike
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([0,1,2,3]).T
    >>> sc.bitwise_xor(a, 3)
    [1 4 1 1]
        3          2          1          0        
    """
    return _pygauss.bitwise_xor(left, right)

def cbrt(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise cube root 

    Parameters
    ----------
    x: ArrayLike 
        Input array expression    

    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation.    

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([0,1,2,3]).T
    >>> sc.cbrt(a)
    [1 4 1 1]
        0.0000     1.0000     1.2599     1.4422     
    """
    return _pygauss.cbrt(x)

def clip(x: ArrayLike, lo: ArrayLike = None, up: ArrayLike = None) -> ShapeletsArray:
    """
    Element-wise, limits the values in an array

    Parameters
    ----------
    x: ArrayLike 
        Input array expression 

    lo: Optional ArrayLike (defaults: None)
        Low values

    up: Optional ArrayLike (defaults: None)
        High values
       
    Returns
    -------
    ShapeletsArray
        A new array with the result of the element-wise operation. 

    Notes
    -----
    The first parameter must resolve to a dimensional array.  Broadcasting 
    rules will be applied to the rest of the parameters. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([0,1,2,3]).T
    >>> sc.clip(a, 1, 2)
    [1 4 1 1]
        0          1          2          3     
    >>> up_vals = sc.array([0,0,1,1]).T
    >>> sc.clip(a, up = up_vals)    
    [1 4 1 1]
        0          0          1          1     
    """
    return _pygauss.clip(x, lo, up)

def conj(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Alias for :obj:`~shapelets.compute.conjugate`
    """
    return conjugate(x)

def conjugate(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes the complex conjugate 

    Parameters
    ----------
    x: ArrayLike 
        Input array expression 

    Returns
    -------
    ShapeletsArray
        A new array with the result of the operation.             
    
    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.conjugate([1-2j, -3+4j])
    [2 1 1 1]
        ( 1.0000, 2.0000) 
        (-3.0000,-4.0000) 

    """
    return _pygauss.conjugate(x)

def complex(real: ArrayLike, img: ArrayLike) -> ShapeletsArray:
    r"""
    Creates a complex array from real and imaginary parts

    Parameters
    ----------
    real: ArrayLike
        A floating point array with the real values.

    img: ArrayLike
        A floating point array with the imaginary values
    
    Returns
    -------
    ShapeletsArray
        A new complex array instance where each component is the 
        element-wise combination of the input arrays.

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> real_part = [1,2,3]
    >>> complex_part = sc.array([6,5,4])
    >>> sc.complex(real_part, complex_part)
    [3 1 1 1]
        (1.0000,6.0000) 
        (2.0000,5.0000) 
        (3.0000,4.0000) 
    >>> sc.complex(real, -1)
    [3 1 1 1]
        (1.0000,-1.0000) 
        (2.0000,-1.0000) 
        (3.0000,-1.0000) 
    >>> sc.complex(real_part, complex_part.T)
    [3 3 1 1]
        (1.0000,6.0000)          (1.0000,5.0000)          (1.0000,4.0000) 
        (2.0000,6.0000)          (2.0000,5.0000)          (2.0000,4.0000) 
        (3.0000,6.0000)          (3.0000,5.0000)          (3.0000,4.0000) 
    """ 
    return _pygauss.complex(real, img)

def divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    r"""
    Element-wise division

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    This operation is equivate to the expression ``left / right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.divide([1,2,3], [3,3,3.]).T
    [1 3 1 1]
        0.3333     0.6667     1.0000 
    >>> sc.divide([1,2,3], 3.0).T
    [1 3 1 1]
        0.3333     0.6667     1.0000     
    >>> sc.divide(3.0, [1,2,3]).T
    [1 3 1 1]
        3.0000     1.5000     1.0000        
    """
    return _pygauss.divide(left, right)

def erf(x: ArrayLike) -> ShapeletsArray:
    r"""
    Applies the error function element-wise

    Parameters
    ----------
    x: ArrayLike
        Input argument to the error function
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation
    
    Notes
    -----
    This function is only defined for real inputs.

    See also
    --------
    erfc

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.reshape(sc.linspace(-3, 3, 10), (2,5))
    >>> sc.erf(x)
    [2 5 1 1]
        -1.0000    -0.9816    -0.3626     0.8427     0.9990 
        -0.9990    -0.8427     0.3626     0.9816     1.0000    

    """
    return _pygauss.erf(x)

def erfc(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Applies the complementary error function element-wise

    Parameters
    ----------
    x: ArrayLike
        Input argument to the complementary error function
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation
    
    Notes
    -----
    This function is only defined for real inputs.

    See also
    --------
    erf

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.reshape(sc.linspace(-3, 3, 10), (2,5))
    >>> sc.erfc(x)
    [2 5 1 1]
        2.0000     1.9816     1.3626     0.1573     0.0010 
        1.9990     1.8427     0.6374     0.0184     0.0000 
            
    """
    return _pygauss.erfc(x)

def exp(y: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes ``e`` to the power ``y``

    Parameters
    ----------
    y: ArrayLike
        Input array expression with the exponents.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the elment-wise operation
    
    See also
    --------
    exp2
        Computes ``2 ** y``
    exp1m
        Computes ``exp(y) - 1``

    Notes
    -----
    This function is equivalent to ``np.e ** y``

    """
    return _pygauss.exp(y)

def exp2(y: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes ``2`` to the power ``y``

    Parameters
    ----------
    y: ArrayLike
        Input array expression with the exponents.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the elment-wise operation
    
    See also
    --------
    exp
        Computes ``np.e ** y``
    exp1m
        Computes ``exp(y) - 1``

    Notes
    -----
    This function is equivalent to ``2 ** y``.
    """
    return _pygauss.exp2(y)

def expm1(y: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes ``exp(y) - 1`` element-wise

    Parameters
    ----------
    y: ArrayLike
        Input array expression with the exponents.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the elment-wise operation
    
    See also
    --------
    exp
        Computes ``np.e ** y``
    exp2
        Computes ``2 ** y``
    log1p
        The inverse of this function.

    Notes
    -----
    This function provides greater precision than ``exp(x) - 1`` for small values of x.
    
    """
    return _pygauss.expm1(y)

def fabs(array_like: ArrayLike) -> ShapeletsArray:  
    r"""
    Compute the absolute values element-wise.

    This function handles complex numbers; therefore, it produces the 
    same results as :obj:`~shapelets.compute.absolute` function.

    Parameters
    ----------
    x: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.fabs([1.0, -1.0, 1+1j, 1-1j]).T
    [1 4 1 1]
        1.0000     1.0000     1.4142     1.4142
    >>> sc.absolute([1.0, -1.0, 1+1j, 1-1j]).T        
    [1 4 1 1]
        1.0000     1.0000     1.4142     1.4142 
    """
    return _pygauss.fabs(array_like)

def factorial(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Computes the factorial function element-wise.

    The factorial function is defined in terms of the Gamma function, :math:`\Gamma(x+1) = x!`

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    tgamma
        Gamma function

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.linspace(0, 4, 8, False).T
    >>> sc.factorial(x)
    [1 8 1 1]
        1.0000     0.8862     1.0000     1.3293     2.0000     3.3234     6.0000    11.6317
    """
    return _pygauss.factorial(x)

def floor_divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Return the largest integer smaller or equal to the element-wise division of the inputs.

    Parameters
    ----------
    left: ArrayLike
        Input array expression as numerator

    right: ArrayLike
        Input array expression as denominator

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation  
    
    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    This operation is equivalent to ``left // right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4,5], dtype = "float32").T
    >>> sc.true_divide(a, 5)
    [1 5 1 1]
        0.2000     0.4000     0.6000     0.8000     1.0000  
    >>> sc.floor_divide(a, 5)
    [1 5 1 1]
        0.0000     0.0000     0.0000     0.0000     1.0000 
    >>> a // 5
    [1 5 1 1]
        0.2000     0.4000     0.6000     0.8000     1.0000           

    """
    return _pygauss.floor_divide(left, right)

def imag(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Returns the imaginary part of complex numbers

    Parameters
    ----------
    x: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation.
    
    Notes
    -----
    If the input array is not complex, this method will return 0s.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.imag([1.0, -1.0, 1+1j, 1-1j]).T
    [1 4 1 1]
        0.0000     0.0000     1.0000    -1.0000
    """
    return _pygauss.imag(x)

def left_shift(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise left shift operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression denoting the number of positions to shift to the left.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    This function is equivalent to ``left << right``. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.left_shift([128, 64, 32], [3,3,3]).T
    [1 3 1 1]
        1024        512        256
        
    >>> a = sc.array([128, 64, 32])
    >>> a << 3
    [3 1 1 1]
        1024 
        512 
        256         
    """
    return _pygauss.left_shift(left, right)

def lgamma(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise application of the natural logarithm of the gamma function.

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    tgamma
        Gamma function

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.linspace(1, 4, 8, True).T
    >>> x.display()
    [1 8 1 1]
        1.0000     1.4286     1.8571     2.2857     2.7143     3.1429     3.5714     4.0000    
    >>> sc.tgamma(x)
    [1 8 1 1]
        1.0000     0.8861     0.9478     1.1568     1.5624     2.2909     3.5988     6.0000    
    >>> sc.log(sc.tgamma(x))
    [1 8 1 1]
        0.0000    -0.1210    -0.0536     0.1457     0.4462     0.8289     1.2806     1.7918     
    >>> sc.lgamma(x)
    [1 8 1 1]
        0.0000    -0.1210    -0.0536     0.1457     0.4462     0.8289     1.2806     1.7918     
    """    
    return _pygauss.lgamma(x)

def log(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise application of the natural logarithm function.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    log2
    log1p
    log10

    """
    return _pygauss.log(x)

def log10(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise application of the base 10 logarithm function.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    log2
    log1p
    log
    """
    return _pygauss.log10(x)

def log1p(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Calculates ``log(1 + x)``.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    log2
    log10
    log

    """
    return _pygauss.log1p(x)

def log2(array_like: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise application of the base 10 logarithm function.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    log1p
    log10
    log
    """
    return _pygauss.log2(array_like)

def mod(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise modulus operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    This operation is equivate to the expression ``left % right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.mod([1,2,3], [3,3,3]).T
    [1 3 1 1]
        1          2          0  
    >>> sc.mod([1,2,3], 3.0).T
    [1 3 1 1]
        1.0000     2.0000     0.0000     
    >>> sc.mod(3.0, [1,2,3]).T
    [1 3 1 1]
        0.0000     1.0000     0.0000  
    """
    return _pygauss.mod(left, right)

def multiply(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise multiply operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    This operation is equivate to the expression ``left * right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.multiply([1,2,3], [3,3,3]).T
    [1 3 1 1]
        3          6          9  
    >>> sc.multiply([1,2,3], 3.0).T
    [1 3 1 1]
        3.0000     6.0000     9.0000     
    >>> sc.multiply([1,2,3], sc.transpose([1,2,3]))
    [3 3 1 1]
        1          2          3 
        2          4          6 
        3          6          9      
    """
    return _pygauss.multiply(left, right)

def negative(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Numerical negative, element-wise.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    Equivalent to the unary ``-`` operator.

    """
    return _pygauss.negative(x)

def positive(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Numerical positive, element-wise.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    This function does nothing but returning a copy of the input array.
    """
    return _pygauss.positive(x)

def power(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise exponential operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    This operation is equivate to the expression ``left ** right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.power([1,2,3], [3,3,3]).T
    [1 3 1 1]
        1          8         27  
    >>> sc.power([1,2,3], 3.0).T
    [1 3 1 1]
        1.0000     8.0000    27.0000     
    >>> sc.power([1,2,3], sc.transpose([1,2,3]))
    [3 3 1 1]
        1          1          1 
        2          4          8 
        3          9         27          
    """
    return _pygauss.power(left, right)

def real(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Returns the real part of complex numbers

    Parameters
    ----------
    x: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.real([1.0, -1.0, 1+1j, 1-1j]).T
    [1 4 1 1]
        1.0000     -1.0000     1.0000    1.0000    
    """
    return _pygauss.real(x)

def reciprocal(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the reciprocal.

    Parameters
    ----------
    x: ArrayLike
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.array([1.0, -2.0, 1+1j, 1-1j]).T
    >>> a = sc.reciprocal(x)
    >>> x * a
    [1 4 1 1]
        (1.0000,0.0000)          (1.0000,0.0000)          (1.0000,0.0000)          (1.0000,0.0000)    
    """
    return _pygauss.reciprocal(x) 

def remainder(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise remainder operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays.  

    Complex arrays are not supported

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.remainder([1,2,3], [3,3,3]).T
    [1 3 1 1]
        1          2          0  
    >>> sc.remainder([1,2,3], 3.0).T
    [1 3 1 1]
        1.0000    -1.0000     0.0000     
    >>> sc.remainder(3.0, [1,2,3]).T
    [1 3 1 1]
        0.0000    -1.0000     0.0000      
    """
    return _pygauss.rem(left, right) 

def right_shift(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise right shift operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    This function is equivalent to ``left >> right``. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.right_shift([1024,512,256], [3,3,3]).T
    [1 3 1 1]
        128         64         32  
    >>> a = sc.array([1024,512,256])
    >>> a >> 3
    [3 1 1 1]
        128 
         64 
         32      
    """
    return _pygauss.right_shift(left, right)

def root(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise n-th root operation

    Parameters
    ----------
    left: ArrayLike
        Input array expression with the nth root

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> roots = sc.iota(5).T + 1
    >>> sc.root(roots, 100)
    [1 5 1 1]
        100.0000    10.0000     4.6416     3.1623     2.5119    
    """
    return _pygauss.root(left, right)

def rsqrt(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Computes the reciprocal or inverse of the square root.

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.rsqrt(x)

def sigmoid(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise application of the sigmoid function, defined as :math:`S(x)=\frac{1}{1+e^{-x}}`

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.sigmoid(x)

def sign(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise indication of the sign of numbers.

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.sign(x)

def signbit(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Tests if values are less than zero.

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.signbit(x)

def sqrt(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise squared root

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.sqrt(x)

def square(x: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise square operation

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    """
    return _pygauss.square(x)

def substract(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    r"""
    Element-wise substraction

    Parameters
    ----------
    left: ArrayLike
        Input array expression 

    right: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    This operation is equivalent to ``left - right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> sc.substract([1,2,3,4,5], 5).T
    [1 5 1 1]
        -4         -3         -2         -1          0       
    """
    return _pygauss.substract(left, right)

def tgamma(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise application of the gamma function.

    Parameters
    ----------
    x: ArrayLike 
        Input array expression
    
    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    See also
    --------
    lgamma
        Gamma function
    factorial
        Computes the factorial using the gamma function.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> x = sc.linspace(1, 4, 8, True).T
    >>> x.display()
    [1 8 1 1]
        1.0000     1.4286     1.8571     2.2857     2.7143     3.1429     3.5714     4.0000    
    >>> sc.tgamma(x)
    [1 8 1 1]
        1.0000     0.8861     0.9478     1.1568     1.5624     2.2909     3.5988     6.0000    
    """        
    return _pygauss.tgamma(x)

def true_divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise true division operation

    Parameters
    ----------
    left: ArrayLike
        The dividend

    right: ArrayLike
        The divisor

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Notes
    -----
    At least one of the parameters must resolve to a dimensional array.  Broadcasting 
    rules will be applied to adjust the size of the arrays. 

    This operation is equivalent to ``left / right``

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1,2,3,4,5], dtype = "float32").T
    >>> sc.true_divide(a, 5)
    [1 5 1 1]
        0.2000     0.4000     0.6000     0.8000     1.0000  
    >>> sc.floor_divide(a, 5)
    [1 5 1 1]
        0.0000     0.0000     0.0000     0.0000     1.0000 
    >>> a / 5
    [1 5 1 1]
        0.2000     0.4000     0.6000     0.8000     1.0000     
    """
    return _pygauss.true_divide(left, right)

def round(x: ArrayLike, decimals: int = 0) -> ShapeletsArray: 
    r"""
    Evenly round to the given number of decimals

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    decimals: int (default: 0)
        Number of decimals.

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1 , 2, 3]).T / 3.0
    >>> a[:,1] *= -1.0 
    >>> a.display(8)
    [1 3 1 1]
        0.33333331    -0.66666663     0.99999994 
    >>> sc.round(a, 2)
    [1 3 1 1]
        0.3300    -0.6700     1.0000     
    """
    return _pygauss.round(x, decimals)

def rint(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1 , 2, 3]).T / 3.0
    >>> a[:,1] *= -1.0 
    >>> a.display(8)
    [1 3 1 1]
        0.33333331    -0.66666663     0.99999994 
    >>> sc.rint(a)
    [1 3 1 1]
        0.0000    -1.0000     1.0000         
    """
    return _pygauss.rint(x)

def fix(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Round to nearest integer towards zero.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1 , 2, 3]).T / 3.0
    >>> a[:,1] *= -1.0 
    >>> a.display(8)
    [1 3 1 1]
        0.33333331    -0.66666663     0.99999994 
    >>> sc.fix(a)
    [1 3 1 1]
        0.0000    -0.0000     0.0000         
    """
    return _pygauss.fix(x)

def floor(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Return the floor of the input, element-wise.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1 , 2, 3]).T / 3.0
    >>> a[:,1] *= -1.0 
    >>> a.display(8)
    [1 3 1 1]
        0.33333331    -0.66666663     0.99999994 
    >>> sc.floor(a)
    [1 3 1 1]
        0.0000    -1.0000     0.0000     
    """
    return _pygauss.floor(x)

def ceil(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Return the ceiling of the input, element-wise.

    Parameters
    ----------
    x: ArrayLike
        Input array expression

    Returns
    -------
    ShapeletsArray
        A new array instance with the results of the element-wise operation

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([1 , 2, 3]).T / 3.0
    >>> a[:,1] *= -1.0 
    >>> a.display(8)
    [1 3 1 1]
        0.33333331    -0.66666663     0.99999994 
    >>> sc.ceil(a)
    [1 3 1 1]
        1.0000    -0.0000     1.0000 
    """
    return _pygauss.ceil(x)

def trunc(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Return the truncated value of the input, element-wise.

    Parameters
    ----------
    x: ArrayLike 
        Input array.

    Returns
    -------
    ShapeletsArray
        A new array instance with all decimal parts are set to zero.

    See also
    --------
    ceil
    floor
    rint

    """
    return _pygauss.trunc(x)

def absolute(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Calculate the absolute value element-wise

    This operation is equivalent to call the builtin function `abs`.

    Parameters
    ----------
    x: ArrayLike 
        Input array.

    See also
    --------
    fabs

    Returns
    -------
    A tensor, with the same dimensions as x, containing the absolute value of each 
    element. For complex value tensors, where each element has the form :math:`a + bi`,
    the absolute value is computed as :math:`\sqrt{a^{2} + b^{2}}`.

    """
    return _pygauss.absolute(x)

def angle(x: ArrayLike, deg: bool = False) -> ShapeletsArray: 
    r"""
    Calculates the angle, or phase, of a tensor of complex 
    numbers element-wise.

    Parameters
    ----------
    x: ArrayLike
        A tensor of complex values.
    deg: Boolean.  Defaults to False.
        Flag to determine if the angles are to be returned in radians (default), or 
        if the output should be angles in degrees.

    Returns
    -------
    A tensor, with the same dimensions as ``x``, where each element is the phase.  
    For 0 complex numbers, that is, those in the form :math:`0+0j`, this method 
    will return the angle as 0.

    """
    return _pygauss.angle(x) if deg == False else _pygauss.angle_deg(x)

def arccos(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse cosine.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arccos`` function is defined as :math:`arccos(z) = \frac{\pi}{2} + i*log(iz+\sqrt{1-z^2})`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse cosine function to each input element.
    """
    return _pygauss.arccos(x)

def arcsin(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse sine.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arcsin`` function is defined as :math:`arcsin(z) = -i*log(i*z+\sqrt{1-z^2})`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse sine function to each input element.
    """    
    return _pygauss.arcsin(x)

def arctan(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse tangent.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arctan`` function is defined as :math:`arctan(z) = \frac{i*log(1-i*z)-log(1+i*z)}{2}`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse tangent function to each input element.
    """    
    return _pygauss.arctan(x)

def arccosh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse hyperbolic cosine function.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arccosh`` function is defined as :math:`arccosh(z) = log(z + \sqrt{z+1}*\sqrt{z-1})`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse hyperbolic cosine function to each 
        input element.
    """  
    return _pygauss.arccosh(x)

def arcsinh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse hyperbolic sine function.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arcsinh`` function is defined as :math:`arcsinh(z) = log(z + \sqrt{z^2+1})`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse hyperbolic sine function to each 
        input element.
    """        
    return _pygauss.arcsinh(x)

def arctanh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise computation of the inverse hyperbolic tangent.

    In addition to floating numbers, this function operates on complex tensors too, 
    where the ``arctanh`` function is defined as :math:`arctanh(z) = \frac{i*log(1+z)-log(1-z)}{2}`

    Parameters
    ----------
    x: ArrayLike
        A tensor of real or complex values.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of applying the inverse hyperbolic tangent function to each 
        input element.
    """       
    return _pygauss.arctanh(x)

def cos(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes cosine function element-wise

    For complex values, ``cos`` is defined as :math:`cos(a+bi) = cos(a)*cosh(b) + sin(a)*sinh(b)*i`

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing cosine to each input element.
    """    
    return _pygauss.cos(x)

def cosh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes hyperbolic cosine element-wise.

    For complex tensor, ``cosh`` is defined as :math:`cosh(a+bi) = cosh(a) * cos(b) + sinh(a)*sin(b)*i`

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing hyperbolic cosine to each input element.

    """    
    return _pygauss.cosh(x)

def degrees(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Convert angles from radians to degrees.

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of the conversion from radians to degrees.

    See also
    --------
    rad2deg
        Alias for the same function.
    """
    return _pygauss.degrees(x)

def rad2deg(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Convert angles from radians to degrees.

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of the conversion from radians to degrees.

    See also
    --------
    degrees
        Alias for the same function.
    """

    return _pygauss.rad2deg(x)

def deg2rad(x: ArrayLike) -> ShapeletsArray: 
    """
    Converts angles from degrees to radians

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of the conversion from degrees to radians.

    See also
    --------
    radians
        Alias for the same function
    """
    return _pygauss.deg2grad(x) 

def radians(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Converts angles from degrees to radians

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of the conversion from degrees to radians.

    See also
    --------
    deg2rad
        Alias for the same function

    """
    return _pygauss.radians(x)

def sin(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes sine function element-wise

    For complex values, ``sin`` is defined as :math:`sin(a+bi) = sin(a)*cosh(b) + cos(a)*sinh(b)*i`

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing sine to each input element.

    """
    return _pygauss.sin(x)

def sinh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes hyperbolic sine element-wise.

    For complex tensor, ``sinh`` is defined as :math:`sinh(a+bi) = sinh(a) * cos(b) + cosh(a)*sin(b)*i`

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing hyperbolic sine to each input element.

    """        
    return _pygauss.sinh(x)

def tan(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes tangent function element-wise

    Complex and float values are supported.

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray 
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing tangent to each input element.
    """
    return _pygauss.tan(x)

def tanh(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Computes hyperbolic tangent element-wise.

    Complex and float values are supported.

    Parameters
    ----------
    x: ArrayLike
        Input array.

    Returns
    -------
    ShapeletsArray
        A tensor of the same dimensions as ``x``, whose elements are the 
        result of computing hyperbolic tangent to each input element.

    """
    return _pygauss.tanh(x)

def hypot(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    r"""
    Computes the length of the hypotenuse of two arrays.  

    Parameters
    ----------
    left: ArrayLike
        Length of one of the sides.

    right: ArrayLike
        Length of the other side.

    Returns
    -------
    ShapeletsArray
        A new array instance.
    
    Notes
    -----
    At least one of the two parameters must be an array expression.  Broadcasting rules
    applies to the operation. 

    Only non complex data is supported.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([1,2,3,4], dtype = "float32")
    >>> sc.hypot(a, 3)
    [4 1 1 1]
        3.1623 
        3.6056 
        4.2426 
        5.0000 
    >>> sc.hypot(a, a.T)
    [4 4 1 1]
        1.4142     2.2361     3.1623     4.1231 
        2.2361     2.8284     3.6056     4.4721 
        3.1623     3.6056     4.2426     5.0000 
        4.1231     4.4721     5.0000     5.6569     
    """
    return _pygauss.hypot(left, right)

def arctan2(y: ArrayLike, x: ArrayLike) -> ShapeletsArray: 
    r"""
    Element-wise arctan operation over the ratio between y and x.

    Parameters
    ----------
    y: ArrayLike
        numerator values

    x: ArrayLike
        denominator values

    Returns
    -------
    ShapeletsArray
        A new array instance.
    
    Notes
    -----
    At least one of the two parameters must be an array expression.  Broadcasting rules
    applies to the operation. 
    """
    return _pygauss.arctan2(x, y)

__all__ = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "hypot", "arctan2", "degrees", 
    "rad2deg", "radians", "deg2rad", "sinh", "cosh", "tanh", "arcsinh", "arccosh", 
    "arctanh", "trunc", "floor", "ceil", "rint", "fix", "round", "exp", "expm1", "exp2", 
    "log", "log10", "log2", "log1p", "signbit", "add", "reciprocal", "positive", "negative", 
    "multiply", "divide", "true_divide", "power", "substract", "floor_divide", "mod", "remainder", 
    "real", "imag", "conj", "conjugate", "complex", "angle", "complex", "sqrt", 
    "cbrt", "square", "absolute", "fabs", "sign", "clip", "sigmoid", "erf", "erfc", "rsqrt", 
    "factorial", "tgamma", "lgamma", "root", "bitwise_and", "bitwise_or", "bitwise_xor", 
    "left_shift", "right_shift" 
]