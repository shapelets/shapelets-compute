from __future__ import annotations
from typing import NamedTuple, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from .__basic_typing import ArrayLike, _ScalarLike
from ._array_obj import ShapeletsArray
from . import _pygauss

AnyScalar = _ScalarLike
FloatOrComplex = Union[complex, float]
ConvDomain = Literal['auto', 'frequency', 'spatial']
ConvMode = Literal['default', 'expand']
NormType = Literal['euclid','lpq','matrix','matrixinf','vector1','vector2','vectorinf','vectorp']
MatMulOptions = Literal['none', 'transpose', 'conjtrans']

def __pygauss_norm_type(tpe: NormType):
    if tpe == 'euclid':
        return _pygauss.NormType.Euclid
    elif tpe == 'lpq':
        return _pygauss.NormType.LPQ
    elif tpe == 'matrix':
        return _pygauss.NormType.Matrix
    elif tpe == 'matrixinf':
        return _pygauss.NormType.MatrixInf
    elif tpe == 'vector1':
        return _pygauss.NormType.Vector1
    elif tpe == 'vector2':
        return _pygauss.NormType.Vector2
    elif tpe == 'vectorinf':
        return _pygauss.NormType.VectorInf
    elif tpe == 'vectorp':
        return _pygauss.NormType.VectorP
    else:
        raise ValueError("Unknown norm type")

def __pygauss_mat_mul_options(opt: MatMulOptions):
    if opt == 'none':
        return _pygauss.MatrixProperties.Default
    elif opt == 'transpose':
        return _pygauss.MatrixProperties.Transposed
    elif opt == 'conjtrans':
        return _pygauss.MatrixProperties.ConjugatedTransposed
    else:
        raise ValueError("Unknown matrix multiplication option")

def __pygauss_conv_domain(domain: ConvDomain):
    if domain=='auto':
        return _pygauss.ConvDomain.Auto
    elif domain == 'frequency':
        return _pygauss.ConvDomain.Frequency
    elif domain == 'spatial':
        return _pygauss.ConvDomain.Spatial
    else:
        raise ValueError("Unknown convolve domain")

def __pygauss_conv_mode(mode: ConvMode):
    if mode == 'default':
        return _pygauss.ConvMode.Default
    elif mode == 'expand':
        return _pygauss.ConvMode.Expand 
    else:
        raise ValueError("Unknown convolve mode")


class EigenResult(NamedTuple):
    values: ShapeletsArray
    """Access to the eigenvalues"""
    vectors: ShapeletsArray
    """Access to the eigenvectors"""

def eigvalsh(data: ArrayLike) -> ShapeletsArray:
    """
    Computes eigenvalues for selfadjoint matrices.
    
    This method operates over floating and complex matrices of 32 and 64 bits.  If 
    presented with an (signed or unsigned) integer or ``float16`` matrix, a
    conversion will occur internally.

    Parameters
    ----------
    data: ArrayLike 
        Squared matrix (nxn)
    
    Returns
    -------
    ShapeletsArray
        Floating (not complex) array with eigenvalues in increasing order.
    """    
    return _pygauss.eigvalsh(data)

def eigvals(data: ArrayLike) -> ShapeletsArray:
    """
    Computes the eigenvalues of general matrices.

    This method operates over floating and complex matrices of 32 and 64 bits.  If 
    presented with an (signed or unsigned) integer or ``float16`` matrix, a
    conversion will occur internally.

    Parameters
    ----------
    data: ArrayLike 
        Squared matrix (nxn)

    Returns
    -------
    ShapeletsArray
        Complex array with the eigenvalues.

    """
    return _pygauss.eigvals(data)

def eigh(data: ArrayLike) -> EigenResult:
    """
    Computes eigenvalues and eigenvectors for selfadjoint matrices.
    
    This method operates over floating and complex matrices of 32 and 64 bits.  If 
    presented with an (signed or unsigned) integer or ``float16`` matrix, a
    conversion will occur internally.

    Parameters
    ----------
    data: ArrayLike 
        Squared matrix (nxn)
    
    Returns
    -------
    EigenResult
        Named tuple with eigen values and vectors.  

    Notes
    -----
    Eigenvalues will be always real and the result will present them in increasing order.  

    """
    return EigenResult(*_pygauss.eigh(data))

def eig(data: ArrayLike) -> EigenResult: 
    """
    Computes eigenvalues and eigenvectors of general matrices.

    This method operates over floating and complex matrices of 32 and 64 bits.  If 
    presented with an (signed or unsigned) integer or ``float16`` matrix, a
    conversion will occur internally.

    Parameters
    ----------
    data: ArrayLike 
        Squared matrix (nxn)

    Returns
    -------
    EigenResult
        Named tuple with eigen values and vectors.  Both values and vectors 
        are returned as complex arrays, matching the bit size of the input 
        array.

    Examples
    --------
    Compute the eigen vectors and values of a 3x3 matrix:

    >>> import shapelets.compute as sc
    >>> m = sc.random.randn((3,3))
    >>> eval, evec = sc.eigen(m)
    >>> eval
    [3 1 1 1]
            (-1.1138, 1.3809) 
            (-1.1138,-1.3809) 
            (-0.2371, 0.0000)     
    >>> evec
    [3 3 1 1]
            (-0.4381,-0.2015)          (-0.4381, 0.2015)          (-0.0597,0.0000) 
            ( 0.1361, 0.0528)          ( 0.1361,-0.0528)          (-0.9269,0.0000) 
            (-0.4076, 0.7616)          (-0.4076,-0.7616)          (-0.3706,0.0000)     
    

    Internal data conversion:

    >>> import shapelets.compute as sc
    >>> m = sc.ones((3,3), "int32")
    >>> eval, _ = sc.eigen(m)
    >>> eval
    [3 1 1 1]
            (-0.0000,0.0000) 
            ( 3.0000,0.0000) 
            ( 0.0000,0.0000) 
    """
    return EigenResult(*_pygauss.eig(data))

def convolve(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray: 
    """
    Returns the discrete, linear convolution of sequences up to 3 dimensions.

    Parameters
    ----------
    signal: ArrayLike
        Input signals.  
    filter: ArrayLike
        Input filters
    mode: ConvMode.
        When mode is set to ``expand`` the output will be ``n+m-1``, where ``n`` is the length of the signal and ``m`` is the 
        length of the filter.
    domain: ConvMode.
        Forces the operation to be executed through a FFT transformation. 

    Returns
    -------
    ShapeletsArray
        A new array holding the results of the convolution operation.  The actual dimensions of the resulting 
        array depends on batching (see notes) and mode parameters.  

    Notes
    -----
    This function delegates the actual operation to the specialized 1d, 2d or 3d convolve operations based on the 
    dimensionality of the signal and filter arrays.  The dimensionality of the operation is computed as the minimum 
    rank of the signal and filter parameters.

    When multiple convolve operations are required, either one signal to many filters, many signals to one filter or,
    alternatively, mutliple signals to multiple filters, use the specialized version and adjust the dimensions 
    of signal and filter to benefit from the implicit parallelism.

    See Also
    --------
    convolve1   
        For 1D operations
    convolve2   
        For 2D operations
    convolve3   
        For 3D operations
    reorder     
        To quickly reorganise the input parameters and adjust their dimensionality.

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__signal__func__convolve.htm>`_

    """
    return _pygauss.convolve(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve1(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray: 
    """
    Returns the discrete, linear convolution of one-dimensional sequences.

    Parameters
    ----------
    signal: ArrayLike
        Input signals.  
    filter: ArrayLike
        Input filters
    mode: ConvMode.
        When mode is set to ``expand`` the output will be ``n+m-1``, where ``n`` is the length of the signal and ``m`` is the 
        length of the filter.
    domain: ConvMode.
        Forces the operation to be executed through a FFT transformation. 

    Returns
    -------
    ShapeletsArray
        A new array holding the results of the convolution operation.  The actual dimensions of the resulting 
        array depends on batching (see notes) and mode parameters.  

    Notes
    -----

    This operation supports batching, that is, the capability to execute multiple convolution operations simultaneously over multiple 
    input vectors, providing the information is organised as follows:

    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+
    | Signal Shape | Filter Shape | Output Shape | Batch Mode              | Description                                                        |
    +==============+==============+==============+=========================+====================================================================+
    | (m,1,1,1)    | (m,1,1,1)    | (m,1,1,1)    | No Batch                | Output will be a single convolved array                            |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+
    | (m,1,1,1)    | (m,n,1,1)    | (m,n,1,1)    | Filter is Batched       | n filters applied to same input                                    |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+
    | (m,n,1,1)    | (m,1,1,1)    | (m,n,1,1)    | Signal is Batched       | 1 filter applied to n inputs                                       |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+
    | (m,n,p,q)    | (m,n,p,q)    | (m,n,p,q)    | Identical Batches       | n*p*q filters applied to n*p*q inputs in one-to-one correspondence |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+
    | (m,n,1,1)    | (m,1,p,q)    | (m,n,p,q)    | Non-overlapping batches | p*q filters applied to n inputs to produce n x p x q results       |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------------+            

    Examples
    --------
    Simple application of a filter that adds the current element and its previous value.

    >>> import shapelets.compute as sc
    >>> r = sc.convolve1([0,1,2,3,4,5,6,7,8,9], [0,1,1])
    >>> print(r.T)
    [1 10 1 1]
    0.0000     1.0000     3.0000     5.0000     7.0000     9.0000    11.0000    13.0000    15.0000    17.0000 

    No batching, one to one convolve operation differences between 'expand' and 'default' modes

    >>> import shapelets.compute as sc
    >>> signal = sc.random.randn(1000)
    >>> filter = sc.random.randn(10)
    >>> sc.convolve1(signal, filter).shape
    (1000, 1)
    >>> sc.convolve1(signal, filter, 'expand').shape
    (1009, 1)

    Batched many signals (10 signals of 1000 elements) against one filter

    >>> import shapelets.compute as sc
    >>> signals = sc.random.randn((1000, 10))
    >>> filter = sc.random.randn(10)
    >>> sc.convolve1(signals, filter).shape
    (1000, 10)

    Batched many signals (10 signals of 1000 elements) against many filters (20 filters of 100 elements)

    >>> import shapelets.compute as sc
    >>> signals = sc.random.randn((1000, 10))
    >>> filters = sc.random.randn((100, 1, 20))
    >>> sc.convolve1(signals, filters).shape
    (1000, 10, 20)

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__signal__func__convolve1.htm>`_
    """
    return _pygauss.convolve1(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve2(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray:
    """
    Returns the discrete, linear convolution of two-dimensional sequences.

    Parameters
    ----------
    signal: ArrayLike
        Input signals.  
    filter: ArrayLike
        Input filters
    mode: ConvMode.
        When mode is set to ``expand`` the output will be ``n+m-1``, where ``n`` is the length of the signal and ``m`` is the 
        length of the filter.
    domain: ConvMode.
        Forces the operation to be executed through a FFT transformation. 

    Returns
    -------
    ShapeletsArray
        A new array holding the results of the convolution operation.  The actual dimensions of the resulting 
        array depends on batching (see notes) and mode parameters.  

    Notes
    -----
    This operation supports batching, that is, the capability to execute multiple convolution operations simultaneously over multiple 
    input vectors, providing the information is organised as follows:

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__signal__func__convolve2.htm>`_
    """
    return _pygauss.convolve2(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve3(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray:
    """
    Returns the discrete, linear convolution of three-dimensional sequences.

    Parameters
    ----------
    signal: ArrayLike
        Input signals.  
    filter: ArrayLike
        Input filters
    mode: ConvMode.
        When mode is set to ``expand`` the output will be ``n+m-1``, where ``n`` is the length of the signal and ``m`` is the 
        length of the filter.
    domain: ConvMode.
        Forces the operation to be executed through a FFT transformation. 

    Returns
    -------
    ShapeletsArray
        A new array holding the results of the convolution operation.  The actual dimensions of the resulting 
        array depends on batching (see notes) and mode parameters.  

    Notes
    -----
    This operation supports batching, that is, the capability to execute multiple convolution operations simultaneously over multiple 
    input vectors, providing the information is organised as follows:

    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------+
    | Signal Shape | Filter Shape | Output Shape | Batch Mode              | Description                                                  |
    +==============+==============+==============+=========================+==============================================================+
    | (m,n,p,1)    | (a,b,c,1)    | (m,n,p,1)    | No Batch                | Output will be a single convolved array                      |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------+
    | (m,n,p,1)    | (a,b,c,d)    | (m,n,p,d)    | Filter is Batched       | d filters applied to same input                              |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------+
    | (m,n,p,q)    | (a,b,c,1)    | (m,n,p,q)    | Signal is Batched       | 1 filter applied to q inputs                                 |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------+
    | (m,n,p,k)    | (a,b,c,k)    | (m,n,p,k)    | Identical Batches       | k filters applied to k inputs in one-to-one correspondence   |
    +--------------+--------------+--------------+-------------------------+--------------------------------------------------------------+

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__signal__func__convolve3.htm>`_    
    """
    return _pygauss.convolve3(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def cholesky(x: ArrayLike, is_upper: bool = True) -> ShapeletsArray:
    """
    Performs a Cholesky decomposition.

    Parameters
    ----------
    x: 2D array
        Input array.  It must be positive definite matrix.
    is_upper: bool, defaults to True
        Indicates if the output should be upper or lower triangular

    Notes
    -----
    Given a `positive definite <https://en.wikipedia.org/wiki/Positive-definite_matrix>`_ matrix :math:`X`, Cholesky decomposition
    computes a decomposition :math:`X = L * U` in such way that :math:`L = U^T`

    Examples
    --------
    Peforming a decomposition

    >>> import shapelets.compute as sc
    >>> x = [[2.+0.j, -0.-3.j], [0.+3.j, 5.+0.j]]
    >>> sc.cholesky(x)
    [2 2 1 1]
            (1.4142,0.0000)          (-0.0000,-2.1213) 
            (0.0000,0.0000)          (0.7071,0.0000)     
    >>> sc.cholesky(x, False)
    [2 2 1 1]
            (1.4142,0.0000)          (0.0000,0.0000) 
            (0.0000,2.1213)          (0.7071,0.0000)     

    """
    return _pygauss.cholesky(x, is_upper)

def det(x: ArrayLike) -> FloatOrComplex:
    """
    Computes the determinant of an input matrix

    Parameters
    ----------
    x: ArrayLike
        Input array

    Returns
    -------
    float or complex
        A scalar value (depending on the type of the input matrix)

    Examples
    --------
    Compute the determinant of a 2D matrix:

    >>> import shapelets.compute as sc
    >>> sc.det([[1, 2], [3, 4]])
    -2.0

    Using a complex matrix:

    >>> import shapelets.compute as sc
    >>> sc.det([[1,2,3j],[4j,5,6],[7,8j,9]])
    (129-321j)

    """
    return _pygauss.det(x)

def dot(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> ShapeletsArray:
    """
    Computes the dot product, also known as inner product, of vectors.

    Parameters
    ----------
    lhs: 1D vector
        Input vector, it must be a columnar vector.

    rhs: 1D vector
        Input vector, it must be a columnar vector.

    conj_lhs: bool, defaults to False
        When true, the inner product will be computed with the conjugate of the ``lhs`` parameter.
    
    conj_rhs: bool, defaults to False
        When true, the inner product will be computed with the conjugate of the ``rhs`` parameter.

    Returns
    -------
    ShapeletsArray
        This method always returns an array.

    See Also
    --------
    dot_scalar
        Executes the same operation but returns a scalar value.

    Examples
    --------

    >>> import shapelets.compute as sc
    >>> sc.dot([2j, 3j], [2j, 3j])           
    [1 1 1 1]
             (-13.0000,0.0000) 
    >>> sc.dot([2j, 3j], [2j, 3j], True, False)
    [1 1 1 1]
             (13.0000,0.0000)     
    """
    return _pygauss.dot(lhs,rhs,conj_lhs,conj_rhs)

def dot_scalar(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> FloatOrComplex: 
    """
    Computes the dot product, also known as inner product, of vectors.

    Parameters
    ----------
    lhs: 1D vector
        Input vector, it must be a columnar vector.

    rhs: 1D vector
        Input vector, it must be a columnar vector.

    conj_lhs: bool, defaults to False
        When true, the inner product will be computed with the conjugate of the ``lhs`` parameter.
    
    conj_rhs: bool, defaults to False
        When true, the inner product will be computed with the conjugate of the ``rhs`` parameter.

    Returns
    -------
    Float or complex value
        
    See Also
    --------
    dot
        For a version of the same functionality but leaving the result in device's memory.

    """    
    return _pygauss.dot_scalar(lhs,rhs,conj_lhs,conj_rhs)


def gemm(a: ArrayLike, b: ArrayLike, c: Optional[ArrayLike] = None, alpha: float = 1.0, beta: float = 0.0, trans_a: bool = False, trans_b: bool = False) -> ShapeletsArray:
    r"""
    Access GEMM Blas Level 3 general matrix multiply.

    This operation is defined as: :math:`C = \alpha * opA(A)opB(B) + \beta * C`, where :math:`\alpha` 
    and :math:`\beta` are both scalars, and :math:`opA` and :math:`opB` are the optional transpose 
    operation before the operation executes.
    
    This operation expects the arrays to be consistent in relation to the element types of A, B and C.

    Parameters
    ----------
    a: 2D array
        Input array
    b: 2D array
        Input array
    alpha: float
        The alpha value 
    beta: float
        The beta value
    c: Optional, 2D array
        Input / Output array
    trans_a: bool, defaults to False
        When set, it transposes ``a`` before GEMM
    trans_b: bool, defaults to False
        When set, it transposes ``b`` before GEMM

    Notes
    -----
    This operation supports batch mode if, at least, either :math:`A` or :math:`B` have more than two dimensions; however,
    the parameters ``alpha`` and ``beta`` will be used for all batches.

    See Also
    --------
    matmul
        For a simple, less terse version of the same operation.

    Returns
    -------
    ShapeletsArray
        Result of GEMM operation

    Examples
    --------
    Simple example of using GEMM

    >>> import shapelets.compute as sc
    >>> a = sc.random.randn((5,5), dtype="float32")
    >>> b = sc.random.randn((5,5), dtype="float32")
    >>> alpha = 0.1
    >>> sc.gemm(a, b, None, alpha)
    [5 5 1 1]
        -0.2978     0.2178    -0.1756     0.0745    -0.2571 
         0.1324    -0.0535     0.1575    -0.0205     0.3111 
         0.2681    -0.3266    -0.1164    -0.2178    -0.0926 
        -0.1185     0.1376    -0.0778     0.0822     0.2401 
        -0.1444    -0.0076    -0.0989    -0.0827    -0.4234     
    >>> c = sc.random.randn((5,5), dtype="float32")
    >>> beta = 0.4
    >>> sc.gemm(a, b, c, alpha, beta)
    [5 5 1 1]
         0.4803    -0.3218    -0.6793     0.1306     0.2275 
        -0.6295     0.4380    -0.7495    -0.0569     0.2767 
         0.5940    -0.7455     0.1700    -0.0880    -0.2894 
        -0.1918     0.2941    -1.1007    -0.3546    -0.2993 
        -0.2061    -0.0331     0.1955     0.2680    -0.9807     

    """    
    return _pygauss.gemm(a, b , c, alpha, beta, trans_a, trans_b)

def inv(array_like: ArrayLike) -> ShapeletsArray:
    """
    Computes the multiplicative inverse of non singular matrices.

    The inverse of a matrix :math:`A`, denoted as :math:`A^{-1}, is a matrix that satisfies :math:`AA^{-1}=I`

    Parameters
    ----------
    array_like: ArrayLike
        Input matrix

    Returns
    -------
    ShapeletsArray

    See Also
    --------
    pinv
        Pseudo-inverse 

    Example
    -------
    >>> import shapelets.compute as sc 
    >>> a = sc.array([[1.,2],[3,4]])
    >>> inv_a = sc.inv(a)
    >>> a @ inv_a
    [2 2 1 1]
        1.0000     0.0000 
        0.0000     1.0000 

    """        
    return _pygauss.inverse(array_like, _pygauss.MatrixProperties.Default)

def lu(x: ArrayLike) -> tuple:
    """
    Perform a LU decomposition.

    A LU decomposition converts an input matrix A intro a lower and upper triangle matrices such that :math:`A = L*U`

    Parameters
    ----------
    x: ArrayLike
        Input matrix

    Returns
    -------
    Tuple
        It returns three ShapeletsArray matrices, lower, upper and a pivot matrix.  See notes.

    Notes
    -----
    For stability reasons, a permutation array is also returned.  To reconstruct the original array
    the permutation matrix is used to index the results of the decomposition, as the following example shows:

    >>> import shapelets.compute as sc
    >>> a = sc.array([[0, 3, 6], [1, 4, 7], [2, 5, 8.]], "float32)
    >>> l, u, p = sc.lu(a)
    >>> l.display()
    [3 3 1 1]
        1.0000     0.0000     0.0000 
        0.0000     1.0000     0.0000 
        0.5000     0.5000     1.0000 
    >>> u.display()
    [3 3 1 1]
        2.0000     5.0000     8.0000 
        0.0000     3.0000     6.0000 
        0.0000     0.0000     0.0000             
    >>> p.display()
    [3 1 1 1]
            2 
            0 
            1 
    >>> m[p, ...] = sc.matmul(l, u)
    >>> m.display()
    [3 3 1 1]
        0.0000     3.0000     6.0000 
        1.0000     4.0000     7.0000 
        2.0000     5.0000     8.0000     

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__lapack__factor__func__lu.htm>`_ 
    """        
    return _pygauss.lu(x)

def matmul(lhs: ArrayLike, rhs: ArrayLike, lhs_options: MatMulOptions = 'none', rhs_options: MatMulOptions = 'none') -> ShapeletsArray:
    """
    Matrix Multiplication

    Parameters
    ----------
    lhs: 2D Array, MxK 
        Input Array
    rhs: 2D Array, KxN
        Input Array
    lhs_options: MatMulOptions, defaults to 'none'
        Applies a transformation to ``lhs`` without further allocation of device or host memory.

    rhs_options: MatMulOptions, defaults to 'none'
        Applies a transformation to ``rhs`` without further allocation of device or host memory.

    Notes
    -----
    This function supports batching.  The following table outlines the different shape arrangements to organise 
    a matrix multiplication:

    +-----------+-----------+--------------+
    | Lhs Shape | Rhs Shape | Output Shape |
    +===========+===========+==============+
    | (M,K,1,1) | (K,N,1,1) | (M,N,1,1)    |
    +-----------+-----------+--------------+
    | (M,K,a,b) | (K,N,a,b) | (M,N,a,b)    |
    +-----------+-----------+--------------+
    | (M,K,1,1) | (K,N,a,b) | (M,N,a,b)    |
    +-----------+-----------+--------------+
    | (M,K,a,b) | (K,N,1,1) | (M,N,a,b)    |
    +-----------+-----------+--------------+

    The last two entries in the table, the 2D matrix is broadcasted to match the dimensions of the other array; this
    operation doesn't alocate any additional memory.

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__blas__func__matmul.htm>`_ 
    """
    return _pygauss.matmul(lhs, rhs,__pygauss_mat_mul_options(lhs_options),__pygauss_mat_mul_options(rhs_options))

def matmulNT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray:
    """
    Executes matrix multiplication :math:`L*R^T`

    See also
    --------
    matmul
        See this function for description of parameters and batching opportunities
    """        
    return _pygauss.matmulNT(lhs, rhs)

def matmulTN(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray: 
    """
    Executes matrix multiplication :math:`L^T*R`

    See also
    --------
    matmul
        See this function for description of parameters and batching opportunities
    """        
    return _pygauss.matmulTN(lhs, rhs)

def matmulTT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray: 
    """
    Executes matrix multiplication :math:`L^T*R^T`

    See also
    --------
    matmul
        See this function for description of parameters and batching opportunities
    """
    return _pygauss.matmulTT(lhs, rhs)   

def matmul_chain(*args) -> ShapeletsArray: 
    """
    """        
    return _pygauss.matmul_chain(*args)

def norm(x: ArrayLike, type: NormType = 'euclid', p: float = 1.0, q: float = 1.0) -> float:
    r"""
    Computes the norm of a vector.

    Given a n-dimensional vector :math:`x`, the vector norm :math:`\left | \textbf{x} \right |_p` for ``p`` = 1,2,... is defined 
    as :math:`\left | \textbf{x} \right |_p = \left ( \sum_i \left | x_i \right |^p \right )^{\frac{1}{p}}`

    Parameters
    ----------
    x: ArrayLike
        n-dimensional vector

    type: NormType (default: 'euclid')
        See notes for norm types.

    p: float (default: 1.0)
        See norm type table in notes

    q: float (default: 1.0)
        See norm type table in notes

    Returns
    -------
    Float value
        Computed norm

    Notes
    -----
    The following norm types and parameters can be specified:

    +-----------+---------------------------------------------------------------------+
    | Norm Type | Description                                                         |
    +===========+=====================================================================+
    | euclid    | Euclidian norm.  Same as vector 2                                   |
    +-----------+---------------------------------------------------------------------+
    | lpq       | General norm computation with arbitrary p and q parameters          |
    +-----------+---------------------------------------------------------------------+
    | matrix    | return the max of column sums                                       |
    +-----------+---------------------------------------------------------------------+
    | matrixinf | return the max of row sums                                          |
    +-----------+---------------------------------------------------------------------+
    | vector1   | treats the input as a vector and returns the sum of absolute values |
    +-----------+---------------------------------------------------------------------+
    | vector2   | treats the input as a vector and returns euclidean norm             |
    +-----------+---------------------------------------------------------------------+
    | vectorinf | treats the input as a vector and returns the max of absolute values |
    +-----------+---------------------------------------------------------------------+
    | vectorp   | treats the input as a vector and returns the p-norm                 |
    +-----------+---------------------------------------------------------------------+

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[1.,2], [3, 5]])
    >>> sc.norm(a)
    6.244997998398398
    >>> sc.norm(a, 'matrix')
    7.0
    >>> sc.norm(a, 'matrixinf')  
    8.0
    >>> sc.norm(a, 'vector1') 
    11.0
    >>> sc.norm(a, 'vector2') 
    6.244997998398398
    >>> sc.norm(a, 'vectorinf') 
    5.0
    >>> sc.norm(a, 'vectorp', p=3.5) 
    5.28157131911548
    >>> sc.norm(a, 'lpq', p=2.0, q=3.0)
    7.214996469470583
    """      
    return _pygauss.norm(x,__pygauss_norm_type(type) ,p, q)

def pinv(x: ArrayLike, tol: float = 1e-06) -> ShapeletsArray: 
    """
    Computes Moore-Penrose pseudoinverse of a matrix.

    A pseudoinverse of a matrix :math:`A` is defined as :math:`A^{+}` such as :math:`AA^{+}A= A`.

    Parameters
    ----------
    x: 2D Array.  MxN
        Input array.  See notes regarding batching.

    tol: float. Defaults to 1e-06
        Lower threshold for finding singular values using :obj:`~shapelets.compute.svd`

    Returns
    -------
    ShapeletsArray
        If the input is MxN, the resulting matrix will be of size NxM

    Notes
    -----
    This operation can be batched if the input array has more than 2 dimensions.  Each MxN slice along the third 
    dimension will have its own pseudoinverse.

    Examples
    --------
    >>> import shapelets.compute as sc
    >>> a = sc.array([[0,3],[1,4],[2,5]], dtype="float32")
    >>> inv = sc.pinv(a)
    >>> a @ inv @ a
    [3 2 1 1]
        0.0000     3.0000 
        1.0000     4.0000 
        2.0000     5.0000 

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__lapack__ops__func__pinv.htm>`_ 
    """        
    return _pygauss.pinverse(x, tol)

def qr(x: ArrayLike) -> tuple: 
    r"""
    Performs QR decomposition

    A QR decomposition transforms an input matrix, :math:`A`, into an orthogonal matrix :math:`Q` 
    and an upper triangular matrix :math:`R` in such way that :math:`A = Q * R` and :math:`Q*Q^T = I`,
    where :math:`I` is the identity matrix.

    Parameters
    ----------
    x: 2D array with shape M,N
        Input array.

    Returns
    -------
    Tuple of ShapeletsArray
        The decomposition returns a tuple of three matrices: :math:`Q`, :math:`R` and :math:`\tau`.
    
    Examples
    --------
    The following examples show the operation:

    >>> import shapelets.compute as sc
    >>> a = sc.array([[0,3],[1,4],[2,5]], dtype="float32")
    >>> q,r,t = sc.qr(a)
    >>> a.same_as(q@r)
    True
    >>> sc.matmulNT(q,q)
    [3 3 1 1]
         1.0000    -0.0000    -0.0000 
        -0.0000     1.0000     0.0000 
        -0.0000     0.0000     1.0000     

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__lapack__factor__func__qr.htm>`_ 
    """        
    return _pygauss.qr(x)

def rank(x: ArrayLike, tol: float = 1e-05) -> int: 
    """
    Finds the rank of the input matrix

    The rank of a matrix is defined as the dimension of the vector space generated by its columns, which corresponds to 
    the maximum number of linear independent columns in the matrix.

    Parameters
    ----------
    x: 2D Array.
        Input array
    tol: float, defaults to 1e-05
        Tolerance value to guide the implementation, which uses :obj:`~shapelets.compute.qr` to decompose the input 
        matrix and find its rank.

    Returns
    -------
    Int
        The maximum number of linearly independent columns in ``x``.

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__lapack__ops__func__rank.htm>`_ 

    Examples
    --------
    Find the rank of a 3D matrix:

    >>> import shapelets.compute as sc
    >>> a = sc.array([[1, 2, 1], [-2, -3, 1], [3, 5, 0]], dtype="float32")
    >>> sc.rank(a)
    2

    Decomposing the same array using :obj:`~shapelets.compute.qr` clearly shows the rank is 2

    >>> q,r,t = sc.qr(a)
    >>> r
    [3 3 1 1]
        -3.7417    -6.1470     0.2673 
         0.0000    -0.4629    -1.3887 
         0.0000     0.0000    -0.0000 

    """     
    return _pygauss.rank(x, tol)   

def svd(x: ArrayLike) -> tuple: 
    """
    Returns the singular value decomposition of a matrix

    A singular value decomposition is a factorization of a matrix :math:`A` into two unitary 
    matrices :math:`U` and :math:`V^T` and a diagonal matrix :math:`S` in such way that 
    :math:`A = USV^T`.

    Parameters
    ----------
    x: 2D Array.  MxN
        Input array
    
    Returns
    -------
    Tuple
        This method returns a tuple of three arrays: :math:`U` (MxM), :math:`S` and :math:`V^T` (NxN).  Please note
        that :math:`S` will be returned as a columnar vector whose elements are the main diagonal of matrix MxN.
    
    Examples
    --------
    The following example shows the usage of ``svd``:

    >>> import shapelets.compute as sc
    >>> a = sc.random.randn((5,5), "float32")
    >>> u, s, vt = sc.svd(a)
    >>> sdiag = sc.diag(s, False)
    >>> a.same_as(u @ sdiag @ vt)
    True

    References
    ----------
    [1] `ArrayFire Documentation <https://arrayfire.org/docs/group__lapack__factor__func__svd.htm>`_ 
    """        
    return _pygauss.svd(x)   


__all__ = [
    "cholesky", "det", "dot", "dot_scalar", "gemm", "inv", "lu", "matmul", "matmulNT", "matmulTN", 
    "matmulTT", "matmul_chain", "norm", "pinv", "qr", "rank", "svd", "convolve", 
    "convolve1", "convolve2", "convolve3", "eig","eigh", "eigvals", "eigvalsh",
    "NormType", "ConvMode", "ConvDomain"
]

