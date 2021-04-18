from __future__ import annotations
from typing import Optional, Union, Literal
from .__basic_typing import ArrayLike, _ScalarLike
from ._array_obj import ShapeletsArray
from . import _pygauss

AnyScalar = _ScalarLike
FloatOrComplex = Union[complex, float]
ConvDomain = Literal['auto', 'frequency', 'spatial']
ConvMode = Literal['default', 'expand']
NormType = Literal['euclid','lpq','matrix','matrixinf','singular','vector1','vector2','vectorinf','vectorp']
MatMulOptions = Literal['none', 'transpose', 'conjtrans']

def __pygauss_norm_type(tpe: NormType):
    if tpe == 'euclid':
        return _pygauss.NormType.euclid
    elif tpe == 'lpq':
        return _pygauss.NormType.LPQ
    elif tpe == 'matrix':
        return _pygauss.NormType.Matrix
    elif tpe == 'matrixinf':
        return _pygauss.NormType.MatrixInf
    elif tpe == 'singular':
        return _pygauss.NormType.Singular
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

def convolve(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray: 
    """
    """
    return _pygauss.convolve(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve1(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray: 
    """
    """
    return _pygauss.convolve1(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve2(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray:
    """
    """
    return _pygauss.convolve2(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def convolve3(signal: ArrayLike, filter: ArrayLike, mode: ConvMode = 'default', domain: ConvDomain = 'auto') -> ShapeletsArray:
    """
    """
    return _pygauss.convolve3(signal, filter, __pygauss_conv_mode(mode), __pygauss_conv_domain(domain))

def cholesky(array_like: ArrayLike, is_upper: bool = True) -> ShapeletsArray:
    """
    """
    return _pygauss.cholesky(array_like, is_upper)

def det(array_like: ArrayLike) -> FloatOrComplex:
    """
    """
    return _pygauss.det(array_like)

def dot(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> ShapeletsArray:
    """
    """
    return _pygauss.dot(lhs,rhs,conj_lhs,conj_rhs)

def dot_scalar(lhs: ArrayLike, rhs: ArrayLike, conj_lhs: bool = False, conj_rhs: bool = False) -> FloatOrComplex: 
    """
    """    
    return _pygauss.dot_scalar(lhs,rhs,conj_lhs,conj_rhs)


def gemm(a: ArrayLike, b: ArrayLike, c: Optional[ArrayLike] = None, alpha: float = 1.0, beta: float = 0.0, transA: bool = False, transB: bool = False) -> ShapeletsArray:
    """
    """    
    return _pygauss.dot_scalar(a,b,c,alpha,beta,transA, transB)

def inverse(array_like: ArrayLike) -> ShapeletsArray:
    """
    """        
    return _pygauss.inverse(array_like, _pygauss.MatrixProperties.Default)

def lu(array_like: ArrayLike) -> tuple:
    """
    """        
    return _pygauss.lu(array_like)

def matmul(lhs: ArrayLike, rhs: ArrayLike, lhs_options: MatMulOptions = 'none', rhs_options: MatMulOptions = 'none') -> ShapeletsArray:
    """
    """
    return _pygauss.matmul(lhs, rhs,__pygauss_mat_mul_options(lhs_options),__pygauss_mat_mul_options(rhs_options))

def matmulNT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray:
    """
    """        
    return _pygauss.matmulNT(lhs, rhs)

def matmulTN(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray: 
    """
    """        
    return _pygauss.matmulTN(lhs, rhs)

def matmulTT(lhs: ArrayLike, rhs: ArrayLike) -> ShapeletsArray: 
    """
    """
    return _pygauss.matmulTT(lhs, rhs)   

def matmul_chain(*args) -> ShapeletsArray: 
    """
    """        
    return _pygauss.matmul_chain(*args)

def norm(array_like: ArrayLike, type: NormType = 'vector2', p: float = 1.0, q: float = 1.0) -> float:
    """
    """        
    return _pygauss.norm(array_like,__pygauss_norm_type(type) ,p, q)

def pinverse(array_like: ArrayLike, tol: float = 1e-06) -> ShapeletsArray: 
    """
    """        
    return _pygauss.pinverse(array_like, tol)

def qr(array_like: ArrayLike) -> tuple: 
    """
    """        
    return _pygauss.qr(array_like)

def rank(array_like: ArrayLike, tol: float = 1e-05) -> int: 
    """
    """     
    return _pygauss.rank(array_like, tol)   

def svd(array_like: ArrayLike) -> tuple: 
    """
    """        
    return _pygauss.rank(array_like)   


__all__ = [
    "cholesky", "det", "dot", "dot_scalar", "gemm", "inverse", "lu", "matmul", "matmulNT", "matmulTN", 
    "matmulTT", "matmul_chain", "norm", "pinverse", "qr", "rank", "svd", "convolve", 
    "convolve1", "convolve2", "convolve3",
    "NormType", "ConvMode", "ConvDomain"
]

