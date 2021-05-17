from __future__ import annotations
from ._pygauss import (
    parallel_range, array, ShapeletsArray, ParallelFor
)

from ._device import Backend
from . import _pygauss
from typing import Union, Optional
from .__basic_typing import ArrayLike, DataTypeLike, Shape, ShapeLike

class ShapeletsArray:
    def __copy__(self) -> ShapeletsArray: ...
    def __deepcopy__(self, memo: object) -> ShapeletsArray: ...
    def __getitem__(self, selector: object) -> ShapeletsArray: ...
    def __setitem__(self, selector: object, value: ArrayLike) -> ShapeletsArray: ...
    def __repr__(self) -> str: ...

    def __int__(self) -> int: 
        """
        Scalar conversion to int
        """

    def __float__(self) -> float: 
        """
        Scalar conversion to float
        """

    def __complex__(self) -> complex: 
        """
        Scalar conversion to complex
        """

    def __add__(self, other: ArrayLike) -> ShapeletsArray:
        """
        Element-wise addition between self + other 
        """

    def __and__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise ``bitwise and`` between self & other
        """ 

    def __len__(self) -> int: 
        """
        Returns the number of rows of this array
        """
    def __eq__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise equality test
        """

    def __floordiv__(self, other: ArrayLike) -> ShapeletsArray:
        """
        Element-wise floor division
        """

    def __ge__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise greater than or equal test
        """

    def __gt__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise greater than test
        """

    def __iadd__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise addition
        """
    def __iand__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise ``bitwise and``
        """

    def __ilshift__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise left shift
        """

    def __imod__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise mod operation
        """

    def __imul__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise multiplication
        """

    def __ior__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise ``bitwise or`` operation
        """

    def __ipow__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise power operation
        """

    def __irshift__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise right shift        
        """

    def __isub__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise substraction        
        """

    def __itruediv__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise true division
        """

    def __ifloordiv__(self, other: ArrayLike) -> ShapeletsArray:         
        """
        Implace element-wise floor division
        """

    def __ixor__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Inplace element-wise ``bitwise xor`` operation
        """

    def __le__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise less than or equal test
        """

    def __lshift__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise left shift operation
        """

    def __lt__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise less than test
        """

    def __matmul__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Matrix multiplication (``@`` operator)
        """

    def __mod__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise mod operation
        """

    def __mul__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise multiplication
        """

    def __ne__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise inequality test
        """

    def __neg__(self) -> ShapeletsArray: 
        """
        Element-wise change of sign
        """

    def __or__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise ``bitwise or`` operation
        """

    def __pow__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise power operation.
        """

    def __invert__(self) -> ShapeletsArray: ...

    def __radd__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rand__(self, other: ArrayLike) -> ShapeletsArray: ... 
    def __rlshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rmatmul__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rmod__(self, other: ArrayLike) -> ShapeletsArray: ... 
    def __rmul__(self, other: ArrayLike) -> ShapeletsArray: ... 
    def __ror__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rpow__(self, other: ArrayLike) -> ShapeletsArray: ... 
    def __rrshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rsub__(self, other: ArrayLike) -> ShapeletsArray:  ... 
    def __rtruediv__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rxor__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rfloordiv__(self, other: ArrayLike) -> ShapeletsArray: ...

    def __rshift__(self, other: ArrayLike) -> ShapeletsArray:  
        """
        Element-wise right shift operation
        """

    def __sub__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise substraction
        """

    def __truediv__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise true division
        """
    def __xor__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Element-wise ``bitwise xor`` 
        """

    def astype(self, type: DataTypeLike) -> ShapeletsArray:
        """
        Returns a new array with a different element type
        """

    def display(self, precision: int = 4) -> None: 
        """
        Prints the contents of this array.

        Parameters
        ----------
        precision: integer, defaults to 4
            Number of decimal digits to display each entry.

        Examples
        --------
        >>> import shapelets.compute as sc
        >>> a = sc.array([1.0, 2.0, 3.0])
        >>> a.display()
        [3 1 1 1]
            1.0000 
            2.0000 
            3.0000 
        >>> a.display(2)
        [3 1 1 1]
            1.00 
            2.00 
            3.00 
        """

    def same_as(self, arr_like: ArrayLike, eps: float = 0.0001) -> bool:
        """
        Compares to another array.

        ``nan`` values will be replaced with zero before comparing the two arrays

        Parameters
        ----------
        arr_like: ArrayLike
            The other array.

        eps: float, defaults to 0.0001
            Threshold for numerical comparisons.

        Returns
        -------
        bool
            True if this array and ``arr_like`` have the same elements; false otherwise.

        Examples
        --------

        >>> import shapelets.compute as sc
        >>> a = sc.array([1.0, 2.0, 3.0])
        >>> a.same_as(a)
        True
        >>> a.same_as([1.1, 2.0, 3.0])
        False
        """

    def eval(self) -> None:
        """
        Forces the array to evaluate itself.

        Operations are asynchronous in nature; this method waits until all the data 
        of this array is fully resolved.

        Notes
        -----
        Methods like :obj:`~shapelets.compute.ShapeletsArray.display()` implicetly 
        execute an ``eval`` operation.

        See Also
        --------
        enable_manual_eval
            To control if the operations are executed synchronously or asynchronously 
        """

    @property
    def real(self) -> ShapeletsArray: 
        """
        Returns the real part of a complex valued matrix.

        Returns
        -------
        ShapeletsArray
            A new array only containing the real part of a complex matrix.  When invoked on a 
            non complex matrix, it will return itself.

        Examples
        --------
        >>> import shapelets.compute as sc
        >>> a = sc.array([1+9j, 2-3j])
        >>> a.real
        [2 1 1 1]
            1.0000 
            2.0000 
        """

    @property
    def imag(self) -> ShapeletsArray: 
        """
        Returns the imaginary part of a complex valued matrix.

        Returns
        -------
        ShapeletsArray
            A new array only containing the imaginary part of a complex matrix.  When invoked on a 
            non complex matrix, it will return itself.

        Examples
        --------

        >>> import shapelets.compute as sc
        >>> a = sc.array([1+9j, 2-3j])
        >>> a.imag
        [2 1 1 1]
             9.0000 
            -3.0000 

        """

    @property
    def H(self) -> ShapeletsArray:
        """
        Returns the conjugate transpose of itself

        Examples
        --------
        >>> import shapelets.compute as sc
        >>> a = sc.array([1+9j, 2-3j])
        >>> a.H
        [1 2 1 1]
                (1.0000,-9.0000)          (2.0000,3.0000) 

        """

    @property
    def T(self) -> ShapeletsArray: 
        """
        Returns the transpose of itself

        Examples
        --------
        >>> import shapelets.compute as sc
        >>> a = sc.array([1+9j, 2-3j])
        >>> a.T
        [1 2 1 1]
                (1.0000,9.0000)          (2.0000,-3.0000)        
        """
        
    @property
    def dtype(self) -> DataTypeLike: 
        """
        Returns the matrix type

        To cast a matrix to a different type use the :obj:`~shapelets.compute.ShapeletsArray.astype` method.

        Examples
        --------
        >>> import shapelets.compute as sc
        >>> a = sc.array([1+9j, 2-3j])
        >>> a.dtype
        dtype('complex128')
        >>> b = a.astype('complex64')
        >>> b.dtype
        dtype('complex128')
        """

    @property
    def is_column(self) -> bool: 
        """
        Returns true if the array is a column vector, that is, it has dimensions Nx1
        """

    @property
    def is_empty(self) -> bool: 
        """
        Returns true if the array has no elements.
        """

    @property
    def is_row(self) -> bool: 
        """
        Returns true if the array is a row vector, that is, it has dimensions 1xN
        """

    @property
    def is_single(self) -> bool: 
        """
        Returns true if the array only has one element.

        Single element arrays can be casted to float, integer or complex variables through 
        the special methods :obj:`~shapelets.compute.ShapeletsArray.__float__`,
        :obj:`~shapelets.compute.ShapeletsArray.__int__` and :obj:`~shapelets.compute.ShapeletsArray.__complex__`
        """

    @property
    def is_vector(self) -> bool: 
        """
        Returns true if the matrix is either a row or a column vector.
        """

    @property
    def is_integer(self) -> bool: 
        """
        Returns true if the matrix holds integer numbers.
        """

    @property
    def is_complex(self) -> bool: 
        """
        Return true if the matrix holds complex numbers
        """

    @property
    def is_bool(self) -> bool: 
        """
        Returns true if the matrix holds boolean values
        """

    @property
    def is_floating(self) -> bool: 
        """
        Returns true if the matrix holds either complex or floating point values.
        """

    @property
    def is_half(self) -> bool: 
        """
        Returns true if the matrix holds 16 bit floating point values.
        """

    @property
    def itemsize(self) -> int: 
        """
        Returns the size in bytes of each element in the array.
        """

    @property
    def ndim(self) -> int: 
        """
        Returns the number of dimensions of the array
        """

    @property
    def shape(self) -> Shape: 
        """
        Returns the shape of the array, that is, a tuple with the dimensionality of each axis.
        """

    @property
    def size(self) -> int: 
        """
        Returns the total number of elements held by this array.
        """

    def eval() -> None:
        """
        Forces evaluation of this array
        """

    __array_priority__ = 30
    __hash__ = None


class ParallelFor:
    def __iter__(self) -> ParallelFor: ...
    def __next__(self) -> ParallelFor: ...

def parallel_range(arg: Union[int, slice]) -> ParallelFor: 
    """
    Builds a parallel iterator.
    """
    return _pygauss.parallel_range(arg)

def array(array_like: ArrayLike, shape: Optional[ShapeLike] = None, dtype: Optional[DataTypeLike] = None) -> ShapeletsArray:
    """
    Converts and interprets the input as an array or tensor.

    Possible inputs are native Python constructs like lists and tuples, but also numpy arrays or
    arrow constructs.  Basically, it will process any object that has array semantics either through
    array methods or buffer protocols.

    Parameters
    ----------
    array_like: ArrayLike
        An array, any object exposing the array interface, an object whose __array__ 
        method returns an array, any (nested) sequence or any object implementing 
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

    shape: ShapeLike. Defaults to None
         When shape is set, array_like object will be adjusted to match the given dimensionality.

    dtype: DataTypeLike
         When dtype is not set, the type will be inferred from the actual array_like object

    See Also
    --------
    zeros
        Create an array with all its elements set to zero
    ones
        Create an array initialized with all its elements set to zero
    full
        Create an array with all its values set to an arbitrary constant

    Examples
    --------
    Create a two dimensional array:

    >>> import shapelets.compute as sc
    >>> sc.array([[1,2],[3,4]])
        [2 2 1 1]
            1          2 
            3          4 
    """
    return _pygauss.array(array_like, shape, dtype)


__all__ = [
    "array", "parallel_range", "ShapeletsArray", "ParallelFor"
]
