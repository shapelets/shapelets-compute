from typing import Optional, Tuple, Union, overload
from .__base import Shape, ArrayLike, DataType, Number

def absolute(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Calculate the absolute value element-wise.
    """

def add(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Add arguments element-wise 
    """

def all(array_like: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    """
    Check if all the elements along a specified dimension are true.
    """

def amax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Return the maximum of an array or minimum along an axis, propagating NaNs
    """

def amin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Return the minimum of an array or minimum along an axis, propagating NaNs
    """

def angle(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns the angle in radians
    """

def angle_deg(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns the angle in degrees
    """

def any(array_like: ArrayLike, dim: Optional[int] = None) -> Union[bool, ShapeletsArray]:
    """
    Check if any the elements along a specified dimension are true.
    """

def arccos(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric inverse cosine element-wise
    """

def arccosh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Inverse hyperbolic cosine, element-wise.
    """

def arcsin(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric inverse sine element-wise
    """

def arcsinh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Inverse hyperbolic sine, element-wise.
    """
def arctan(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric inverse tangent element-wise
    """
def arctan2(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise arc tangent of the inputs
    """

def arctanh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Inverse hyperbolic tangent, element-wise.
    """

def argmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """
    Returns the indices and values of the maximum values along an axis, with NaNs propagated.
    """

def argmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """
    BLAB LBABLABLABLA 
    Returns the indices and values of the minimum values along an axis, propagating NaNs
    """

def array(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
        Converts and interprets the input as an array or tensor.

        Possible inputs are native Python constructs like lists and tuples, but also numpy arrays or
        arrow constructs.  Basically, it will process any object that has array semantics either through
        array methods or buffer protocols.

        Parameters
        ----------
        array_like: ArrayLike construct
        shape: Int or Tuple of ints. Defaults to None
             When shape is set, array_like object will be adjusted to match the given dimensionality.
        dtype: A compatible expression numpy dtype.
             When dtype is not set, the type will be inferred from the actual array_like object

        Examples
        --------
        Create a two dimensional array:

        >>> import shapelets.compute as sh
        >>> a = sh.array([[1,2],[3,4]])

        
    """
@overload
def batch() -> ScopedBatch: ...
@overload
def batch(arg0: function) -> object: ...

def bitwise_and(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise bitwise and.
    """

def bitwise_or(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise bitwise or
    """
def bitwise_xor(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise bitwise xor
    """
def cast(array_like: ArrayLike, dtype: DataType) -> ShapeletsArray:
    """
    Creates a new array by casting the original array
    """
def cbrt(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the cube-root of an array, element-wise.
    """
def ceil(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the ceiling of the input, element-wise.
    """
def cholesky(array_like: ArrayLike, is_upper: bool = True) -> ShapeletsArray:
    """
    Computes the Cholesky decomposition a positive definite matrix.
    The resulting matrix is the triangular matrix of the decomposition; multiply it with its conjugate transpose to reproduce the input matrix.
    """
def clip(array_like: ArrayLike, lo: ArrayLike = None, up: ArrayLike = None, *, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray: ...
    
@overload
def complex(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Builds a complex tensor from a real one.

    Constructs a new complex array two independent sources
    """
@overload
def complex(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray: ...
    
def conj(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Gets the complex conjugate
    """
def conjugate(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Gets the complex conjugate
    """
def convolve(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = ConvMode.Default, domain: ConvDomain = ConvDomain.Auto) -> ShapeletsArray:
    """
    TODO
    """
def convolve1(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = ConvMode.Default, domain: ConvDomain = ConvDomain.Auto) -> ShapeletsArray:
    """
    TODO
    """
def convolve2(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = ConvMode.Default, domain: ConvDomain = ConvDomain.Auto) -> ShapeletsArray:
    """
    TODO
    """
def convolve3(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = ConvMode.Default, domain: ConvDomain = ConvDomain.Auto) -> ShapeletsArray:
    """
    TODO
    """
def corrcoef(a: ShapeletsArray, b: ShapeletsArray) -> Union[float, complex]:
    """
    Computes Pearson product-moment correlation coefficient.
    """
def cos(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric cosine element-wise
    """
def cosh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Hyperbolic cosine, element-wise.
    """
def count_nonzero(array_like: ArrayLike, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Count the number of non zero elements in an array along a specified dimension
    """
def covp(a: ShapeletsArray, b: ShapeletsArray) -> ShapeletsArray:
    """
    Find the covariance (population) of values between two arrays.
    """
def covs(a: ShapeletsArray, b: ShapeletsArray) -> ShapeletsArray:
    """
    Find the covariance (sample) of values between two arrays.
    """
def cumprod(array_like: ArrayLike, dim: int = 0) -> Union[float, complex, ShapeletsArray]:
    """
    Cumulative product of an array along a specified dimension, propagating NaNs
    """
def cumsum(array_like: ArrayLike, dim: int = 0) -> Union[float, complex, ShapeletsArray]:
    """
    Cumulative sum of an array along a specified dimension, propagating NaNs
    """
def deg2rad(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Degrees to radians, element-wise
    """
def degrees(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Radians to degrees, element-wise
    """
def det(array_like: ArrayLike) -> Union[float_, complex]:
    """
    Computes the determinant of a matrix.
    """
def device_gc() -> None:
    """
    Forces a garbage collection on the memory device
    """
def diag(a: ShapeletsArray, index: int = 0, extract: bool = False) -> ShapeletsArray:
    """
    Operates with diagonals
    Using extract parameter one is able to either create a diagonal matrix from a vector (false) or extract a diagonal from a matrix to a vector (true)
    """
def diff1(array_like: ArrayLike, dim: int) -> ShapeletsArray:
    """
    Find the first order differences along specified dimensions
    """
def diff2(array_like: ArrayLike, dim: int) -> ShapeletsArray:
    """
    Find the second order differences along specified dimensions
    """
def divide(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a true division of the inputs, element-wise.
    """
def dot(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> ShapeletsArray:
    """
    Scalar dot product between two vectors.  Also referred to as the inner product.  The result is kept as an array in the device.
    """
def dot_scalar(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> Union[float, complex]:
    """
    Scalar dot product between two vectors.  Also referred to as the inner product.  The result returned as a scalar.
    """
def empty(shape: Shape, dtype: DataType = 'float32') -> ShapeletsArray: ...
    
def enable_manual_eval(new_value: bool) -> None:
    """
    Changes the way results are computed.  
    When manually evaluation is disabled, the system will compute as soon as possible, reducing the opportunities for kernel fusion; however, when manual evaluation is enabled, computations have a better chance of merging, resulting in a far more effective computation.  When enabled, results should be requested through `eval` methods and `sync` to ensure device's work queue is completed.
    """
def equal(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise equality test.  The result is always a boolean tensor
    """
def erf(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Computes the error function value
    """
def erfc(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Complementary Error function value
    """
def eval(*args) -> None:
    """
    Forces the evaluation of all the arrays
    """
def exp(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Calculate the exponential of all elements in the input array
    """
def exp2(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Calculate 2**p for all p in the input array.
    """
def expm1(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Calculate exp(x) - 1 for all elements in the array
    """
def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: DataType = 'float32') -> ShapeletsArray:
    """
        Return a 2-D array with ones on the diagonal and zeros elsewhere.

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
        I : array of shape (N,M)
          An array where all elements are equal to zero, except for the `k`-th
          diagonal, whose values are equal to one.
    """
def fabs(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Calculate the absolute value element-wise.
    """
def factorial(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Factorial function
    """
def fft(signal: ArrayLike, odim: Optional[int] = 0, norm: Optional[float_] = None) -> ShapeletsArray:
    """
    Fast fourier transform on one dimensional signals.
                Parameters
                ----------
                    signal: One dimensional array.  Required
                    odim  : Integer, Defaults to zero.
                            The length of the output signal, in order to truncate or pad the input signal
                    norm  : Float, Defaults to none.
                            The scaling factor; if not provided, it is computed internally.
              
    """
def fix(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Round to nearest integer towards zero.
    """
def flat(array_like: ArrayLike) -> ShapeletsArray:
    """
    It flattens an array to one dimension
    """
def flatnonzero(array_like: ArrayLike) -> ShapeletsArray:
    """
    Return indices that are non-zero in the flattened version of a
    """
def flip(array_like: ArrayLike, dimension: int = 0) -> ShapeletsArray:
    """
    Flips an array along a dimension
    """
def floor(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the floor of the input, element-wise.
    """
def floor_divide(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the largest integer smaller or equal to the division of the inputs.
    """
def fmin(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Minimum of two inputs, ignoring NaNs

    Maximum of two inputs, ignoring NaNs
    """
def full(shape: Shape, fill_value: ArrayLike, dtype: DataType = 'float32') -> ShapeletsArray:
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
def gemm(a: ArrayLike, b: ArrayLike, c: Optional[ArrayLike] = None, alpha: float_ = 1.0, beta: float_ = 0.0, transA: bool = False, transB: bool = False) -> ShapeletsArray:
    """
    Performs a GEMM operation ```C = \alpha * opA(A)opB(B) + \beta * C```
    """
def get_available_backends() -> List[Backend]:
    """
    Returns a list with all available backends in this computer
    """
def get_backend() -> Backend:
    """
            Returns the current backend.

            Please note that all the arrays will be implicitly created in the active backend and
            mixing arrays created with different backends would yield a runtime exception.
        
    """
def get_device() -> DeviceInfo:
    """
    Returns the current or active device
    """
def get_device_memory(dev: Optional[Union[int, DeviceInfo]] = None) -> DeviceMemory:
    """
            Reports the current memory utilization on a particular device.

            When no parameter is provided it will return the memory utilization associated with the
            current device; however, one could either use a ::class::`shapelets.DeviceInfo` or the `id`
            property of the device to report over a different device; when used explicitly, the
            default device will be changed for the duration of the call but, on method termination,
            the default device will be restored.
        
    """
def get_devices() -> List[DeviceInfo]:
    """
    Returns a list of devices found within the active backend.
    """
def greater(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise greater than test.  The result is always a boolean tensor
    """
def greater_equal(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise greater than or equal test.  The result is always a boolean tensor
    """
def has_backend(test_backend: Backend) -> int:
    """
    Checks if a particular backend is supported in this platform.
    """
def hypot(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Given the sides of a triangle, returns the hypotenuse
    """
def identity(shape: Shape, dtype: DataType = 'float32') -> ShapeletsArray:
    """
        Creates an identity array with diagonal values set to one.

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
def ifft(coeff: ArrayLike, odim: Optional[int] = 0, norm: Optional[float_] = None) -> ShapeletsArray:
    """
    Fast fourier transform on one dimensional signals.
                Parameters
                ----------
                    coeff : FFT coefficients.
                    odim  : Integer, Defaults to zero.
                            The length of the output signal, in order to truncate or pad the input signal
                    norm  : Float, Defaults to none.
                            The scaling factor; if not provided, it is computed internally.
              
    """
def imag(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Extracts the imaginary part of a complex array or matrix
    """
def intersect(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray:
    """
    Find the union of two arrays.
    """
def inverse(array_like: ArrayLike, options: MatrixProperties = MatrixProperties.Default) -> ShapeletsArray:
    """
    Computes the inverse.
    """
def iota(shape: Shape, tile: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray:
    """
    Create an sequence [0, shape.elements() - 1] and modify to specified dimensions dims and then tile it according to tile
    """
def iscomplex(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a bool array, where True if input element is complex.
    """
def isfinite(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a boolean tensor where all zero positions are set to True
    """
def isinf(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a boolean tensor where all infinite positions are set to True
    """
def isnan(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a boolean tensor where all nan positions are set to True
    """
def isreal(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a bool array, where True if input element is complex.
    """
def join(lst: list, dimension: int = 0, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
        Joins up to 10 arrays along a particular dimension.

        In the case that not all objects in the lst are arrays, the parameters shape and dtype would guide
        the transformation; if those parameters are not set, the first array in the list will determine
        the shape and type of those entries that are not defined as arrays.
        
    """
def left_shift(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise shift to the left
    """
def less(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise less than test.  The result is always a boolean tensor
    """
def less_equal(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise less than or equal test.  The result is always a boolean tensor
    """
def lgamma(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Logarithm of absolute values of Gamma function
    """
def log(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Natural logarithm
    """
def log10(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Logarithm base 10
    """
def log1p(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Natural logarithm of (1 + in)
    """
def log2(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Logarithm base 2
    """
def logical_and(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Performs element-wise logical and.  The result is always a boolean tensor
    """
def logical_not(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Performs element-wise logical complement.  The result is always a boolean tensor
    """
def logical_or(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Performs element-wise logical or.  The result is always a boolean tensor
    """
def lower(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray:
    """
    Create a lower triangular matrix from input array
    The parameter unit_diag forces the diagonal elements to be one.
    """
def lu(array_like: ArrayLike) -> tuple: ...
    
def manual_eval_enabled() -> bool:
    """
    Informs if computations would only be triggered when a eval is directly requested.
    """
def matmul(lhs: ArrayLike, rhs: ArrayLike, lhs_options: MatrixProperties = MatrixProperties.Default, rhs_options: MatrixProperties = MatrixProperties.Default) -> ShapeletsArray:
    """
    Matrix multiplication with the desired transformations, without taking further memory.
    """
def matmulNT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray:
    """
    Matrix multiplication after performing a transpose on rhs, without taking further memory.
    """
def matmulTN(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray:
    """
    Matrix multiplication after performing a transpose on lhs, without taking further memory.
    """
def matmulTT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray:
    """
    Matrix multiplication after performing a transpose on each one, without taking further memory.
    """
def matmul_chain(*args) -> ShapeletsArray:
    """
    Chains matrix multiplications
    """
def matrixprofile(ta: ShapeletsArray, m: int, tb: Optional[ShapeletsArray] = None) -> MatrixProfile:
    """
    TODO
    """
def matrixprofileLR(ta: ShapeletsArray, m: int) -> dict:
    """
    TODO
    """
def maximum(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Maximum of two inputs, with NaNs propagated.
    """
def mean(a: ShapeletsArray, weights: Optional[ShapeletsArray] = None, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Computes mean on an array.
    When the parameter dim is unset, it computes the mean across all values in the matrix.  When dim has a value, it computes the mean across a particular dimension; if dim is -1, the mean will be produced over the first non trivial dimension.
    The result of this computation is either an array, or a scalar value (complex or float)
    """
def median(a: ShapeletsArray, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Computes median on an array.
    When the parameter dim is unset, it computes the median across all values in the matrix.  When dim has a value, it computes the median across a particular dimension; if dim is -1, the median will be produced over the first non trivial dimension.
    The result of this computation is either an array, or a scalar value (complex or float)
    """
def minimum(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Minimum of two inputs, with NaNs propagated.
    """
def mod(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return element-wise remainder of division.
    """
def moddims(array_like: ArrayLike, shape: Shape) -> ShapeletsArray:
    """
    Changes the dimensions of an array without changing the data
    """
def multiply(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Multiply arguments element-wise.
    """
def nan_to_num(array_like: ArrayLike, nan: float = 0.0, inf: float = 0.0) -> ShapeletsArray:
    """
    Return the minimum of an array or minimum along an axis, propagating NaNs
    """
def nanargmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """
    Returns the indices and values of the maximum values along an axis, with NaNs propagated.
    """
def nanargmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[Tuple[int, float], Tuple[int, complex], Tuple[ShapeletsArray, ShapeletsArray]]:
    """
    Returns the indices and values of the minimum values along an axis, ignoring NaN
    """
def nancumprod(array_like: ArrayLike, dim: int = 0) -> Union[float, complex, ShapeletsArray]:
    """
    Cumulative product of an array along a specified dimension, ignoring NaNs
    """
def nancumsum(array_like: ArrayLike, dim: int = 0) -> Union[float, complex, ShapeletsArray]:
    """
    Cumulative sum of an array along a specified dimension, ignoring NaNs
    """
def nanmax(array_like: ArrayLike, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    The maximum value of an array along a given axis, ignoring any NaNs.
    """
def nanmin(array_like: ArrayLike, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    The minimum value of an array along a given axis, ignoring any NaNs.
    """
def nanscan(array_like: ArrayLike, dim: int = 0, nan: float = 0.0, op: ScanOp = ScanOp.Add, inclusive_scan: bool = True) -> ShapeletsArray:
    """
    Generalized scan of an array, which can the operations defined in ScanOp; this function replaces NaN values with the value provided in parameter `nan`, which defaults to 0.0
    """
def negative(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Numerical negative, element-wise.
    """
def norm(array_like: ArrayLike, type: NormType = NormType.Vector2, p: float = 1.0, q: float = 1.0) -> float: ...
    
def not_equal(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise inequality test.  The result is always a boolean tensor
    """
def ones(shape: Shape, dtype: DataType = 'float32') -> ShapeletsArray:
    """
    Creates an array with the given dimensions with all its elements set to zero.
    """
def pad(array_like: ArrayLike, begin: Shape, end: Shape, fill_type: BorderType) -> ShapeletsArray:
    """
     Pads an array

              Ensure the tuples for begin and end are specific as they represent the increase at the beginning and
              the ending; if you want to add one row at the beginning and one row to the end use (1,0,0,0) and
              (1,0,0,0) as parameters.

              
    """
def parallel_range(arg: Union[int, slice]) -> ParallelFor: ...
    
def pinverse(array_like: ArrayLike, tol: float = 1e-06) -> ShapeletsArray: ...
    
def positive(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Numerical positive, element-wise.
    """
def power(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    First array elements raised to powers from second array, element-wise.
    """
def product(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Calculate the product of all the elements along a specified dimension.

    This function is equivalent to both prod and nanprod in numpy; simply set the value of `nan_value` parameter to either include NaNs in the multiplication (None) or replace NaN values with your choice.
    """
def qr(array_like: ArrayLike) -> tuple: ...
    
def rad2deg(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Radians to degrees, element-wise
    """
def radians(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Degrees to radians, element-wise
    """
def range(shape: Shape, seq_dim: int = -1, dtype: DataType = 'float32') -> ShapeletsArray:
    """
    Creates an array with [0, n] values along the seq_dim which is tiled across other dimensions.
    """
def rank(array_like: ArrayLike, tol: float = 1e-05) -> int: ...
    
def real(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Extracts the real part of a complex array or matrix
    """
def reciprocal(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns the reciprocal of the argument (1/x), element wise
    """
def rem(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return element-wise remainder of division.
    """
def reorder(array_like: ArrayLike, x: int, y: int = 1, z: int = 2, w: int = 3) -> ShapeletsArray:
    """
    It modifies the order of data within an array by exchanging data according to the change in dimensionality. The linear ordering of data within the array is preserved.
    """
def right_shift(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Element-wise shift to the right
    """
def rint(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Round elements of the array to the nearest integer.
    """
def root(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Root function
    """
def round(array_like: ArrayLike, decimals: int = 0, *, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Evenly round to the given number of decimals.
    """
def rsqrt(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    The reciprocal or inverse square root of input arrays (1/sqrt(x))
    """
def scan(array_like: ArrayLike, dim: int = 0, op: ScanOp = ScanOp.Add, inclusive_scan: bool = True) -> ShapeletsArray:
    """
    Generalized scan of an array, which can the operations defined in ScanOp.
    """
def set_backend(arg0: Backend) -> None:
    """
            Changes the active or current backend.

            Please note that all the arrays will be implicitly created in the active backend and
            mixing arrays created with different backends would yield a runtime exception.
        
    """
def set_device(arg0: Union[int, DeviceInfo]) -> bool:
    """
            Changes the current or active device

            To select the new device, one could either use a ::class::`shapelets.DeviceInfo` or the `id`
            property of the device.  If there is being an effective change, this method will return
            true; false otherwise.
        
    """
def shift(array_like: ArrayLike, x: int, y: int = 0, z: int = 0, w: int = 0) -> ShapeletsArray:
    """
    Shifts data in a circular buffer fashion along a chosen dimension
    """
def sigmoid(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Sigmoid function
    """
def sign(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns an element-wise indication of the sign of a number.
    """
def signbit(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns element-wise True where signbit is set (less than zero)
    """
def sin(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric sine element-wise
    """
def sinh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Hyperbolic sine, element-wise.
    """
def sort(array_like: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]:
    """
    Sort the array along a specified dimension.  This method returns a tuple with the data and the indices that would have sort the array (sort and argsort)
    """
def sort_keys(data: ArrayLike, keys: ArrayLike, dim: int = 0, asc: bool = True) -> Tuple[ShapeletsArray, ShapeletsArray]:
    """
    Sort the array along a specified dimension using an auxiliary array containing the indexing keys.  This method returns a tuple with the data and the keys sorted
    """
def sqrt(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the non-negative square-root of an array, element-wise.
    """
def square(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the element-wise square of the input.
    """
def stdev(a: ShapeletsArray, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Computes stdev on an array.
    """
def substract(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Subtract arguments, element-wise.
    """
def sum(array_like: ArrayLike, dim: Optional[int] = None, nan_value: Optional[float] = None) -> Union[float, complex, ShapeletsArray]:
    """
    Calculate the sum of all the elements along a specified dimension.

    This function is equivalent to both sum and nansum in numpy; simply set the value of `nan_value` parameter to either include NaNs in the sumation (None) or replace NaN values with your choice.
    """
def svd(array_like: ArrayLike) -> tuple: ...
    
def sync(dev: Optional[Union[int, DeviceInfo]] = None) -> None:
    """
    Blocks until the device has finished processing.
    """
def tan(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Trigonometric tangent element-wise
    """
def tanh(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Hyperbolic tangent element-wise.
    """
def tgamma(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Gamma function
    """
@overload
def tile(array_like: ArrayLike, dims: Shape) -> ShapeletsArray:
    """
    Repeats an array along the specified dimension

    Repeats an array along the specified dimension
    """
@overload
def tile(array_like: ArrayLike, x: int, y: int = 1, z: int = 1, w: int = 1) -> ShapeletsArray: ...
    
def topk_max(a: ShapeletsArray, k: int) -> tuple:
    """
    The top k max values along a given dimension of the input array
    """
def topk_min(a: ShapeletsArray, k: int) -> tuple:
    """
    The top k max values along a given dimension of the input array
    """
@overload
def transpose(array_like: ArrayLike, dims: bool = False) -> ShapeletsArray:
    """
    Performs a standard matrix transpose
    """
@overload
def transpose(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray: ...
    
def true_divide(left: ArrayLike, right: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Returns a true division of the inputs, element-wise.
    """
def trunc(array_like: ArrayLike, shape: Optional[Shape] = None, dtype: Optional[DataType] = None) -> ShapeletsArray:
    """
    Return the truncated value of the input, element-wise.
    """
def union(x1: ArrayLike, x2: ArrayLike, is_unique: bool = False) -> ShapeletsArray:
    """
    Find the union of two arrays.
    """
def unique(array_like: ArrayLike, is_sorted: bool = False) -> ShapeletsArray:
    """
    Find the unique elements of an array.
    """
def upper(array_like: ArrayLike, unit_diag: bool = False) -> ShapeletsArray:
    """
    Create a upper triangular matrix from input array
    The parameter unit_diag forces the diagonal elements to be one.
    """
def var_p(a: ShapeletsArray, weights: Optional[ShapeletsArray] = None, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]: ...
    
def var_s(a: ShapeletsArray, weights: Optional[ShapeletsArray] = None, dim: Optional[int] = None) -> Union[float, complex, ShapeletsArray]: ...
    
def where(condition: ArrayLike, x: ArrayLike = None, y: ArrayLike = None) -> ShapeletsArray:
    """
    An array with elements from x where condition is True, and elements from y elsewhere.
    """
def zeros(shape: Shape, dtype: DataType = 'float32') -> ShapeletsArray:
    """
    Creates an array with the given dimensions with all its elements set to zero.
    """
