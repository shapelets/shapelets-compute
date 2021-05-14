from __future__ import annotations

from typing import Optional, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from ._array_obj import ShapeletsArray
from .__basic_typing import ArrayLike, ShapeLike
from . import _pygauss

NormType = Literal['backward', 'ortho', 'forward']

def __convertNorm(norm=None):
    if norm is None or (norm == 'backward'):
        return _pygauss.fftNorm.Backward
    elif norm == 'ortho':
        return _pygauss.fftNorm.Ortho
    elif norm == 'forward':
        return _pygauss.fftNorm.Forward
    else:
        return float(norm)

def ifft(c: ArrayLike, shape: Optional[ShapeLike] = None, norm: Optional[Union[NormType,float]] = None) -> ShapeletsArray:
    r"""
    Computes the inverse FFT

    Parameters
    ----------
    c: ArrayLike
        Input FFT coefficients.
    
    shape: ShapeLike, defaults to None
        When not set, the dimensions of of the coefficient input array will determine the geometry of the output.  When set,
        it controls cropping and padding, very much in the same way as the :obj:`~shapelets.compute.fft.fft` operation does.

    norm: NormType or float, defaults to None
        When not explicetely set, it defaults to 'backward'.  See function :obj:`~shapelets.compute.fft.fft` for a list of 
        supported values and their interpretation.
    
    Returns
    -------
    ShapeletsArray
        Results of applying the inverse fourier transform to the input coefficients.  Please note the type of the array 
        will be complex.

    Examples
    --------
    A simple back and forth example with a 100000 size signal:

    >>> import shapelets.compute as sc
    >>> a = sc.random.randn(100000)
    >>> af = sc.fft.fft(a)
    >>> b = sc.fft.ifft(af).real
    >>> a.same_as(b)
    True
    """
    return _pygauss.ifft(c, __convertNorm(norm), shape)

def fft(a: ArrayLike, shape: Optional[ShapeLike] = None, norm: Optional[Union[NormType,float]] = None) -> ShapeletsArray:
    r"""
    Computes a Fourier transform.

    Parameters
    ----------
    a: ArrayLike
        Input array.  It could be a complex or floating array up to three dimensions.
    
    shape: ShapeLike, defaults to None
        When not set, the dimensions of the array will determine the dimensionality and size of the transformation. When 
        this parameter is set, it determines if the input should be cropped or padded with zeros.

    norm: NormType or float
        When not set, it uses 'backward' norm.  Alternatively, it can be set to 'ortho', 'forward' or an explicit value.

    Returns
    -------
    ShapeletsArray
        Complex array with FFT coefficients.

    Notes
    -----
    Norms can be explicitely set to an arbitrary floating number or use one of the following standard scale settings:

    +-----------+----------------------------+----------------------------+
    | Norm Type | To Frequency Domain        | To Time Domain             |
    +===========+============================+============================+
    | backwards | :math:`1.0`                | :math:`\frac{1}{n}`        |
    +-----------+----------------------------+----------------------------+
    | ortho     | :math:`\frac{1}{\sqrt{n}}` | :math:`\frac{1}{\sqrt{n}}` |
    +-----------+----------------------------+----------------------------+
    | forward   | :math:`\frac{1}{n}`        | :math:`1.0`                |
    +-----------+----------------------------+----------------------------+

    .. note:

        FFTs in shapelets can be run up to three dimensions with this function; simply adjust the 
        geometry of the input array and adjust the size of the transformation with the shape parameter.

    See Also
    --------
    fftfreq
        Return the Discrete Fourier Transform sample frequencies

    Examples
    --------
    One dimensional fft:

    >>> import shapelets.compute as sc
    >>> a = sc.random.randn(100)
    >>> af = sc.fft.fft(a)

    The same transformation as before, but controlling padding and norm:

    >>> afp = sc.fft.fft(a, 128, 'ortho')
    >>> afp.shape
    (128, 1)

    Two dimensional fft:

    >>> a = sc.random.randn((100, 100))
    >>> af = sc.fft.fft(a)
    >>> af.shape
    (100, 100)
    """
    return _pygauss.fft(a, __convertNorm(norm), shape)

def rfft(a: ArrayLike, shape: Optional[ShapeLike] = None, norm: Optional[Union[NormType,float]] = None) -> ShapeletsArray:
    r"""
    Returns Fourier Transform for real input

    Parameters
    ----------
    a: ArrayLike
        Input array.
    
    shape: ShapeLike, defaults to None
        When not set, the dimensions of the array will determine the dimensionality and size of the transformation. When 
        this parameter is set, it determines if the input should be cropped or padded with zeros.

    norm: NormType or float
        When not set, it uses 'backward' norm.  Alternatively, it can be set to 'ortho', 'forward' or an explicit value.

    Returns
    -------
    ShapeletsArray
        Coefficients of the Fourier Transform.

    Notes
    -----
    Norms can be explicitely set to an arbitrary floating number or use one of the following standard scale settings:

    +-----------+----------------------------+----------------------------+
    | Norm Type | To Frequency Domain        | To Time Domain             |
    +===========+============================+============================+
    | backwards | :math:`1.0`                | :math:`\frac{1}{n}`        |
    +-----------+----------------------------+----------------------------+
    | ortho     | :math:`\frac{1}{\sqrt{n}}` | :math:`\frac{1}{\sqrt{n}}` |
    +-----------+----------------------------+----------------------------+
    | forward   | :math:`\frac{1}{n}`        | :math:`1.0`                |
    +-----------+----------------------------+----------------------------+

    .. note:

        FFTs in shapelets can be run up to three dimensions with this function; simply adjust the 
        geometry of the input array and adjust the size of the transformation with the shape parameter.

    When the FT is computed for purely real input, the output is Hermitian-symmetric, i.e. the negative 
    frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and 
    the negative-frequency terms are therefore redundant.

    See Also
    --------
    rfftfreq
        Return the Discrete Fourier Transform sample frequencies

    Examples
    --------
    The following example shows the differences between ``fft`` and ``rfft``:

    >>> import shapelets.compute as sc
    >>> sc.fft.rfft([0,1,0,0])
    [3 1 1 1]
         ( 1.0000, 0.0000) 
         ( 0.0000,-1.0000) 
         (-1.0000, 0.0000) 
    >>> sc.fft.fft([0,1,0,0])
    [4 1 1 1]
         ( 1.0000, 0.0000) 
         ( 0.0000,-1.0000) 
         (-1.0000, 0.0000) 
         (-0.0000, 1.0000)     
    """
    return _pygauss.rfft(a, __convertNorm(norm), shape)

def irfft(c: ArrayLike, shape: ShapeLike, norm: Optional[Union[NormType,float]] = None) -> ShapeletsArray:
    r"""
    Computes the inverse FFT for real input

    Parameters
    ----------
    c: ArrayLike
        Input FFT coefficients obtained through :obj::`~shapelets.compute.fft.rfft`
    
    shape: ShapeLike
        Expected dimensionality of the transformation.

    norm: NormType or float, defaults to None
        When not explicetely set, it defaults to 'backward'.  See function :obj:`~shapelets.compute.fft.rfft` for a list of 
        supported values and their interpretation.
    
    Returns
    -------
    ShapeletsArray
        Results of applying the inverse fourier transform to the input coefficients.

    Examples
    --------
    A simple back and forth example with a 100000 size signal:

    >>> import shapelets.compute as sc
    >>> a = sc.random.randn(100000)
    >>> af = sc.fft.rfft(a)
    >>> b = sc.fft.irfft(af, a.shape)
    >>> a.same_as(b)
    True    
    """
    return _pygauss.irfft(c, __convertNorm(norm), shape)

def rfftfreq(n: int, d: float = 1.0) -> ShapeletsArray:
    r"""
    TODO
    """
    return _pygauss.rfftfreq(n, d)

def fftfreq(n: int, d: float = 1.0) -> ShapeletsArray:
    r"""
    Return the Discrete Fourier Transform sample frequencies.

    Parameters
    ----------
    n: int
        Window length
    
    d: float, defaults to 1.0
        Sample spacing

    Returns
    -------
    ShapeletsArray
        Array of length ``n`` with the sample frequencies

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> 

    """
    return _pygauss.fftfreq(n, d)


def spectral_derivative(signal: ArrayLike, kappa_spec: Union[float, ArrayLike] = 1.0, shift: bool = True) -> ShapeletsArray:
    r"""
    Computes the derivative of a signal using spectral (FT) methods.

    Parameters
    ----------
    signal: ArrayLike
        Column vector with the signal 

    kappa_spec: float, defaults to 1.0
        The specification for building the kappa coeffients.  It could be either double number, 
        denoting a domain length to scale the automatic generation of kappa vector from ``-n/2`` to ``n/2``.  
        Alternatively, one can pass a column vector of size n for the desired values.    

    shift: bool, defaults to True
        When kappa_spec is a vector, this flag determines if it is required to adjust the vector values 
        by calling :obj:`~shapelets.compute.fft.fftshift`.

    Returns
    -------
    ShapeletsArray
        Derivative using FFT.

    References
    ----------
    [1] Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control
        Steven L. Brunton
    """
    return _pygauss.spectral_derivative(signal, kappa_spec, shift)


def fftshift(x: ArrayLike, axes: Optional[Union[int, List[int]]] = None) -> ShapeletsArray:
    """
    Shift the zero-frequency component to the center of the spectrum.
    """
    return _pygauss.fftshift(x, axes)

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "rfftfreq",
    "spectral_derivative",
    "fftshift",
    "NormType"
]
