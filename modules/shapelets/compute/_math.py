from __future__ import annotations
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss

def add(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    Adds arguments element-wise
    """
    return _pygauss.add(left, right)

def bitwise_and(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.bitwise_and(left, right)

def bitwise_or(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.bitwise_or(left, right)
    
def bitwise_xor(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.bitwise_xor(left, right)

def cbrt(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.cbrt(array_like)

def clip(array_like: ArrayLike, lo: ArrayLike = None, up: ArrayLike = None) -> ShapeletsArray:
    """
    """
    return _pygauss.clip(array_like, lo, up)

def conj(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.conj(array_like)

def conjugate(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.conjugate(array_like)

def complex(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.complex(left, right)

def divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.divide(left, right)

def erf(array_like: ArrayLike) -> ShapeletsArray:
    """
    """
    return _pygauss.erf(array_like)


def erfc(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.erfc(array_like)

def exp(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.exp(array_like)

def exp2(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.exp2(array_like)

def expm1(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.expm1(array_like)

def fabs(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.fabs(array_like)

def factorial(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.factorial(array_like)

def floor_divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.floor_divide(left, right)

def imag(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.imag(array_like)

def left_shift(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.left_shift(left, right)

def lgamma(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.lgamma(array_like)

def log(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.log(array_like)

def log10(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.log10(array_like)

def log1p(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.log1p(array_like)

def log2(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.log2(array_like)

def mod(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.mod(left, right)

def multiply(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.multiply(left, right)

def negative(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.negative(array_like)

def positive(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.positive(array_like)

def power(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.power(left, right)

def real(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.real(array_like)

def reciprocal(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.reciprocal(array_like) 

def remainder(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.rem(left, right) 

def right_shift(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.right_shift(left, right)

def root(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.root(left, right)

def rsqrt(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.rsqrt(array_like)

def sigmoid(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.sigmoid(array_like)

def sign(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.sign(array_like)

def signbit(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.signbit(array_like)

def sqrt(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.sqrt(array_like)

def square(array_like: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.square(array_like)

def substract(left: ArrayLike, right: ArrayLike) -> ShapeletsArray:  
    """
    """
    return _pygauss.substract(left, right)

def tgamma(array_like: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.tgamma(array_like)

def true_divide(left: ArrayLike, right: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.true_divide(left, right)

def round(array_like: ArrayLike, decimals: int = 0) -> ShapeletsArray: 
    r"""
    TODO
    """
    return _pygauss.round(array_like, decimals)

def rint(array_like: ArrayLike) -> ShapeletsArray: 
    r"""
    TODO
    """
    return _pygauss.rint(array_like)

def fix(array_like: ArrayLike) -> ShapeletsArray: 
    r"""
    TODO
    """
    return _pygauss.fix(array_like)

def floor(array_like: ArrayLike) -> ShapeletsArray: 
    f"""
    TODO
    """
    return _pygauss.floor(array_like)

def ceil(array_like: ArrayLike) -> ShapeletsArray: 
    r"""
    TODO
    """
    return _pygauss.ceil(array_like)

def trunc(array_like: ArrayLike) -> ShapeletsArray: 
    r"""
    TODO
    """
    return _pygauss.trunc(array_like)

def absolute(x: ArrayLike) -> ShapeletsArray: 
    r"""
    Calculate the absolute value element-wise

    This operation is equivalent to call the builtin function `abs`.

    Parameters
    ----------
    x: ArrayLike 
        Input array.

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
        TODO
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
        TODO
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