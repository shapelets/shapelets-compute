from typing import Optional, Tuple, Union, overload
from .__base import ShapeletsArray, DataType, Shape

class RandomEngineType():
    """
    Built-in engines for random number generation

    Members:

      Default : Defaults to Philox

      Mersenne : Mersenne GP 11213.

      Threefry : Threefry 2X32_16.

      Philox : Philox 4x32_10.
    """
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __init__(self, arg0: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    Default: shapelets.compute._pygauss.RandomEngineType # value = <RandomEngineType.Default: 100>
    Mersenne: shapelets.compute._pygauss.RandomEngineType # value = <RandomEngineType.Mersenne: 300>
    Philox: shapelets.compute._pygauss.RandomEngineType # value = <RandomEngineType.Default: 100>
    Threefry: shapelets.compute._pygauss.RandomEngineType # value = <RandomEngineType.Threefry: 200>
    __members__: dict # value = {'Default': <RandomEngineType.Default: 100>, 'Mersenne': <RandomEngineType.Mersenne: 300>, 'Threefry': <RandomEngineType.Threefry: 200>, 'Philox': <RandomEngineType.Default: 100>}

class ShapeletsRandomEngine():
    def beta(self, a: float, b: float, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a Beta distribution.  Alpha is what is called shape or K parameter of the Gamma distribution
        """
    def chisquare(self, df: float, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a chi-square distribution.  df is the degree of freedom
        """
    def exponential(self, scale: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from an exponential distribution.  Scale is the inverse of lambda.
        """
    def gamma(self, alpha: float, scale: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a Gamma distribution.  Alpha is what is called shape or K parameter of the Gamma distribution
        """
    def logistic(self, loc: float = 0.0, scale: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a logistic distribution.
        """
    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a log-normal distribution.
        """
    def multivariate_normal(self, mean: object, cov: object, samples: int, dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a logistic distribution.
        """
    def normal(self, loc: float = 0.0, scale: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw random samples from a normal (Gaussian) distribution.
        """
    def standard_normal(self, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw random samples from a normal (Gaussian) distribution.
        """
    def uniform(self, low: float = 0.0, high: float = 1.0, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Samples are uniformly distributed over the half-open interval [low, high)
        """
    def wald(self, mean: float, scale: float, shape: Shape = (1, 1), dtype: DataType = 'float32') -> ShapeletsArray: 
        """
        Draw samples from a Wald, or inverse Gaussian, distribution.
        """

def default_rng(type: RandomEngineType = RandomEngineType.Default, seed: int = 0) -> ShapeletsRandomEngine:
    """
    Creates a new random engine
    """

def randn(shape: Shape, dtype: DataType = 'float32', engine: typing.Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates a new array using random numbers drawn from a normal distribution
    """

def random(shape: Shape, dtype: DataType = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates a new array using random values drawn from a uniform distribution
    """    

def randint(low: int, high: Optional[int] = None, shape: Shape = (1,1), dtype: DataType = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Return random integers from low (inclusive) to high (exclusive)

    Returns random integers from the discrete uniform distribution of the specified dtype in the half-open
    interval [low, high). If high is None (the default), then results are from [0, low).
    """