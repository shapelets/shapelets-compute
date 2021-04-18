"""
Random module documentation goes here
"""

from __future__ import annotations
from typing import Optional, Literal
from .__basic_typing import ShapeLike, DataTypeLike, ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    default_rng as __default_rng, 
    randint as __randint, 
    randn as __randn, 
    random as __random, 
    permutation as __permutation, 
    ShapeletsRandomEngine as __ShapeletsRandomEngine,
    RandomEngineType as __RandomEngineType
)

class ShapeletsRandomEngine():
    def __init__(self, imp: __ShapeletsRandomEngine) -> None:
        self.__imp = imp

    def beta(self, a: float, b: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO beta
        """
        return self.__imp.beta(a, b, shape, dtype)

    def chisquare(self, df: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO chisquare
        """
        return self.__imp.chisquare(df, shape, dtype)
    
    def exponential(self, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO exponential
        """
        return self.__imp.exponential(scale, shape, dtype)
    
    def gamma(self, alpha: float, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO gamma
        """
        return self.__imp.gamma(scale, alpha, scale, dtype)
    
    def logistic(self, loc: float = 0.0, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO logistic
        """
        return self.__imp.logistic(loc, scale,scale, dtype)
    
    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO lognormal
        """
        return self.__imp.lognormal(mean, sigma, shape, dtype)
    
    def multivariate_normal(self, mean: object, cov: object, samples: int, dtype: DataTypeLike = 'float32') -> ShapeletsArray: 
        """
        TODO multivariate_normal
        """
        return self.__imp.multivariate_normal(mean, cov, samples, dtype)
    
    def normal(self, loc: float = 0.0, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:  
        """
        TODO normal
        """
        return self.__imp.normal(loc, scale, shape, dtype)

    def standard_normal(self, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:  
        """
        TODO standard_normal
        """
        return self.__imp.standard_normal(shape, dtype)

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:  
        """
        TODO uniform
        """
        return self.__imp.uniform(low, high, shape, dtype)

    def wald(self, mean: float, scale: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:  
        """
        TODO wald
        """
        return self.__imp.wald(mean, scale, shape, dtype)

    def permutation(self, x: ArrayLike, axis = 0) -> ShapeletsArray:  
        """
        TODO permutation
        """
        return self.__imp.permutation(x, axis)


RandomEngineType = Literal['mersenne', 'philox', 'threefry', 'default']

def random_engine(type: RandomEngineType = 'default', seed: int = 0) -> ShapeletsRandomEngine:
    """
    Constructs a new instance of a ``ShapeletsRandomEngine``.
    """
    native_ret = __RandomEngineType.Default
    if type == 'mersenne':
        native_ret = __RandomEngineType.Mersenne
    elif type == 'philox':
        native_ret = __RandomEngineType.Philox
    elif type == 'threefry':
        native_ret = __RandomEngineType.Threefry
    elif type != 'default':
        raise ValueError("Unknown random engine type")

    return ShapeletsRandomEngine(__default_rng(native_ret, seed))


def randn(shape: ShapeLike, dtype: DataTypeLike = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates an array whose elements are drawn from the standard normal distribution.

    Parameters
    ----------
    shape: ShapeLike

    dtype: DataTypeLike

    engine: Optional ShapeletsRandomEngine. Defaults to None

    Returns
    -------
    ShapeletsArray

    """
    return __randn(shape, dtype, engine)

def random(shape: ShapeLike, dtype: DataTypeLike = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates an array whose elements are drawn from the continuous uniform distribution in the interval `[0.0, 1.0)` 
    """
    return __random(shape, dtype, engine)

def randint(low: int, high: Optional[int] = None, shape: ShapeLike = (1,1), dtype: DataTypeLike = 'int32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates an array whose elements are integers ranging from `low` (inclusive) to  `high` (exclusive)
    """
    return __randint(low, high, shape, dtype, engine)

def permutation(x: ArrayLike, axis = 0, engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Randomly shuffles an array around a particular axis.
    """
    return __permutation(x, axis, engine)

__all__ = [
    "random_engine",
    "randint", 
    "randn", 
    "random",
    "permutation",
    "ShapeletsRandomEngine",
    "RandomEngineType"
]