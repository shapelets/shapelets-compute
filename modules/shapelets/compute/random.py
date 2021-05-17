from __future__ import annotations
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .__basic_typing import ShapeLike, DataTypeLike, ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    default_rng as __default_rng,
    randint as __randint,
    randn as __randn,
    random as __random,
    permutation as __permutation,
    ShapeletsRandomEngine,
    RandomEngineType as __RandomEngineType
)

class ShapeletsRandomEngine:
    """
    Wrapper class around a random number algorithm suitable to be used in GPU / OpenCL computations

    Use :obj:`~shapelets.compute.random.random_engine` as a constructor function, where it is possible 
    to specify the underlyting random generator algorithm as well as the a seed value for controllable 
    results.
    """

    def beta(self, a: float, b: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the beta distribution.

        The calculation is based on the Gamma distribution, in such way that:

        .. math::

            \\\begin{matrix} X \sim \Gamma(\alpha, \theta) \\ Y \sim \Gamma(\beta, \theta) \end{matrix} \longrightarrow \frac{X}{X+Y} \sim B(\alpha,\beta)

        Parameters
        ----------
        a: float
            It is normally called shape or also known as the `K` parameter of the gamma distribution.        
        b: float, defaults to 1.0
            The scale parameter.
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the beta distribution.   

        """

    def chisquare(self, df: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the chi square distribution.

        The calculation is based on the Gamma distribution, in such way that:

        .. math::

            X \sim \Gamma(\frac{df}{2},2) \longrightarrow \chi^2(df)

        Parameters
        ----------
        df: float
            Degrees of freedom.        
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the chi square distribution.  
        """

    def exponential(self, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the exponential distribution.

        The calculation is based on the Uniform U(0,1) distribution, in such way that:

        .. math::

            X \sim U(0,1) \longrightarrow \frac{-\log(1 - X)}{\lambda} \sim E(\lambda)

        Parameters
        ----------
        scale: float
            Inverse of lambda.        
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the exponential distribution.  
        """

    def gamma(self, alpha: float, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the gamma distribution.

        The implementation is based on the algorithm 4.33 found in reference [1]

        Parameters
        ----------
        alpha: float
            It is normally called shape or also known as the `K` parameter of the gamma distribution.        
        scale: float, defaults to 1.0
            The scale parameter.
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the gamma distribution.            

        References
        ----------
        [1] Base line implementation as defined in: 
            `Handbook of Monte Carlo Methods <http://www.maths.uq.edu.au/~kroese/montecarlohandbook>`_
            D.P. Kroese, T. Taimre, Z.I. Botev. John Wiley & Sons, 2011.

        """

    def logistic(self, loc: float = 0.0, scale: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the logistic distribution.

        The implementation is based on uniform distribution, transforming it the following way:

        .. math::

            X \sim U(0, 1) \longrightarrow \mu + \beta(\log(X) - \log(1 - X)) \sim Logistic(\mu,\beta)

        where ``loc`` represents mu and ``scale`` is represented by beta.

        Parameters
        ----------
        loc: float, defaults to 0.0
            Location parameter
        scale: float, defaults to 1.0
            Scale parameter
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the logistic distribution.
        """

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the lognormal distribution.

        Based on the standard normal distribution, the lognormal is calculated through:

        .. math::

            Z \sim N(0,1) \longrightarrow e^{\mu + \sigma Z} \sim LogNormal(\mu, \sigma)

        Parameters
        ----------
        mean: float, defaults to 0.0
            Expected value (:math:`\mu`)
        sigma: float, defaults to 1.0
            Standard deviation of the variable's natural logaritm (:math:`\sigma`)
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the lognormal distribution.
        """

    def multivariate_normal(self, mean: ArrayLike, cov: ArrayLike, samples: int, dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Draw random samples from a multivariate normal distribution.

        A multivariate normal distribution is specified by its mean and covariance matrix. 

        Parameters
        ----------
        mean: 1-D array of length N
            Mean of each one of the series to generate

        cov: 2-D array of shape NxN
            Symmetric and positive-semidefinite covariance matrix
        
        samples: int
            Determines the lenght of each series to generate.  
        
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements            

        Returns
        -------
        ShapeletsArray
            A new array, of dimensions Nxsamples. 

        """

    def normal(self, mean: float = 0.0, sigma: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Draw random samples from a normal (Gaussian) distribution.

        Parameters
        ----------
        mean: float, defaults to 0.0
            Mean of the distribution

        sigma: float, defaults to 1.0
            Standard deviation of the distribution
        
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
            
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements          

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the gaussian distribution.

        """

    def standard_normal(self, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Draw random samples from a standard normal (Gaussian) distribution.

        A standard gaussian distribution is one with mean 0.0 and 1.0 standard deviation.

        Parameters
        ----------
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
            
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements          

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the gaussian distribution.

        """

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Draw samples from a uniform distribution.

        Parameters
        ----------
        low: float, defaults to 0.0
            Lower bound for the output.  All values will be greater than or equal to low.

        high: float, defaults to 1.0
            Upper bound for the output.  All values will be less than high.

        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
            
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements          

        Returns
        -------
        ShapeletsArray
            A new array instance whose values are drawn from the uniform distribution.
        """

    def wald(self, mean: float, scale: float, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'float32') -> ShapeletsArray:
        r"""
        Return samples from the Wald distribution, also known as the inverse Gaussian distribution.

        Parameters
        ----------
        mean: float, defaults to 0.0
            Expected value 
        scale: float, defaults to 1.0
            Also known as lambda parameter. 
        shape: ShapeLike, defaults to (1,1)
            Dimensions of the tensor.
        dtype: DataTypeLike, defaults to 'float32'
            Type of the resulting elements

        Returns
        -------
        ShapeletsArray
            An new array instance whose values are drawn from the inverse Gaussian distribution.


        """

    def permutation(self, x: ArrayLike, axis=0) -> ShapeletsArray:
        r"""
        Randomly shuffles an array around a particular axis.

        Parameters
        ----------
        x: ArrayLike.  Defaults to (1,1).
            Input array

        axis: int.  Defaults to 0
            Axis to be shuffled

        Returns
        -------
        ShapeletsArray
            A randomly suffled array.
        """


RandomEngineType = Literal['mersenne', 'philox', 'threefry', 'default']


def random_engine(type: RandomEngineType = 'default', seed: int = 0) -> ShapeletsRandomEngine:
    """
    Constructs a new :obj:`~shapelets.compute.random.ShapeletsRandomEngine`.

    Parameters
    ----------
    type: RandomEngineType.  
        Selects the random generator algorithm. 

    seed: int.  Defaults to zero
        Seed value for initialization.

    Returns
    -------
    ShapeletsRandomEngine
        A new instance of a random engine with the selected algorithm and seed.

    Notes
    -------
    There are 3 different engines to choose from, where ``default`` value 
    maps to ``philox`` engine: 

    +----------+-------------------+---------+
    | Engine   | Implementation    | Default |
    +==========+===================+=========+
    | philox   | Philox 4x32-10    |   Yes   |
    +----------+-------------------+---------+
    | threefry | Threefry 2X32-16  |         |
    +----------+-------------------+---------+
    | mersenne | Mersenne GP 11213 |         |
    +----------+-------------------+---------+

    References
    ----------
    [1] Parallel random numbers: As easy as 1, 2, 3.  
        John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw.

        In Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis.
        DOI:`10.1145/2063384.2063405 <https://dl.acm.org/doi/10.1145/2063384.2063405>`_

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

    return __default_rng(native_ret, seed)


def randn(shape: ShapeLike, dtype: DataTypeLike = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates an array whose elements are drawn from the standard normal distribution ``N(0,1)``.

    Parameters
    ----------
    shape: ShapeLike.  Defaults to (1,1).
        Shape of the array to be returned.

    dtype: DataTypeLike.  Defaults to float32.
        Final type of elements of the array.

    engine: ShapeletsRandomEngine.  Defaults to None.
        When not set, uses the default, implicit, random engine algorithm.  However,
        when it is set, this method is equivalent to call 
        :obj:`~shapelets.compute.random.ShapeletsRandomEngine.standard_normal` on the engine.

    Returns
    -------
    ShapeletsArray
        An new array instance whose values are drawn from the standard normal distribution.

    See Also
    --------
    :obj:`~shapelets.compute.random.ShapeletsRandomEngine`

    Examples
    --------
    Three by three array of samples from N(2,9):

    >>> import shapelets.compute as sc
    >>> 2.0 + 3 * sc.random.randn((3,3))
    [3 3 1 1]
    -0.7740     3.0547    -0.3060 
    2.5425     0.9645     2.7238 
    9.6323     2.6574    -1.5845 

    """
    return __randn(shape, dtype, engine)


def random(shape: ShapeLike, dtype: DataTypeLike = 'float32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Creates an array whose elements are drawn from the continuous uniform distribution in the interval `[0.0, 1.0)` 

    Parameters
    ----------
    shape: ShapeLike.  Defaults to (1,1).
        Shape of the array to be returned.

    dtype: DataTypeLike.  Defaults to float32.
        Final type of elements of the array.

    engine: ShapeletsRandomEngine.  Defaults to None.
        When not set, uses the default, implicit, random engine algorithm.  However,
        when it is set, this method is equivalent to call 
        :obj:`~shapelets.compute.random.ShapeletsRandomEngine.uniform` on the engine.

    Returns
    -------
    ShapeletsArray
        An new array instance whose values are drawn from the standard normal distribution.

    Examples
    --------
    A row vector with 5 elements drawn from the uniform distribution:

    >>> import shapelets.compute as sc
    >>> sc.random.random((1,5))
    [1 5 1 1]
        0.1583     0.3712     0.3543     0.6450     0.9675 

    """
    return __random(shape, dtype, engine)


def randint(low: int, high: Optional[int] = None, shape: ShapeLike = (1, 1), dtype: DataTypeLike = 'int32', engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    """
    Return random integers drawn from the “discrete uniform” in the range low (inclusive) to high (exclusive).

    Parameters
    ----------
    low: int
        When `high` parameter is `None`, it indicates the maximum (exclusive) integer for 
        the range `[0, low(`.  

        If `high` is specified, this parameter sets the miminum (inclusive) of the range.  

    high: Optional int.  Defaults to None.
        When set, determines the highest (exclusive) value for the range.

    shape: ShapeLike.  Defaults to (1,1).
        Shape of the array to be returned.

    dtype: DataTypeLike.  Defaults to int32.
        Final type of elements of the array.

    engine: ShapeletsRandomEngine.  Defaults to None.
        When not set, uses the default, implicit, random engine algorithm.  However,
        when it is set, this method is equivalent to call 
        :obj:`~shapelets.compute.random.ShapeletsRandomEngine.uniform` on the engine adjusting
        the parameters and type for the correct result.

    Returns
    -------
    ShapeletsArray
        An new array instance whose values are drawn from the uniform distribution.

    Examples
    --------
    3 by 3 matrix whose elements range from 0 to 10 (exclusive).

    >>> import shapelets.compute as sc
    >>> sc.random.randint(10, shape=(3,3))
    [3 3 1 1]
            6          2          2 
            0          0          3 
            9          5          7    

    Random row vector of 5 elements in the range ``[10,20(``

    >>> import shapelets.compute as sc
    >>> sc.random.randint(10, 20, (1,5))
    [1 5 1 1]
            16         10         19         12         10 

    """
    return __randint(low, high, shape, dtype, engine)


def permutation(x: ArrayLike, axis: int = 0, engine: Optional[ShapeletsRandomEngine] = None) -> ShapeletsArray:
    r"""
    Randomly shuffles an array around a particular axis.

    Parameters
    ----------
    x: ArrayLike.  Defaults to (1,1).
        Input array

    axis: int.  Defaults to 0
        Axis to be shuffled

    engine: ShapeletsRandomEngine.  Defaults to None.
        When not set, uses the default, implicit, random engine algorithm.  However,
        when it is set, this method is equivalent to call 
        :obj:`~shapelets.compute.random.ShapeletsRandomEngine.uniform` on the engine adjusting
        the parameters and type for the correct result.

    Returns
    -------
    ShapeletsArray
        A randomly suffled array.

    Examples
    --------
    Given the following 2 by 5 matrix

    >>> import shapelets.compute as sc
    >>> a = sc.array([[1,2,3,4,5],[6,7,8,9,0]])
    [2 5 1 1]
            1          2          3          4          5 
            6          7          8          9          0 

    One could reshuffle its columns by

    >>> sc.random.permute(x, 1)        
    [2 5 1 1]
            2          5          4          1          3 
            7          0          9          6          8 

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
