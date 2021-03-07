from typing import Optional, Tuple, Union, overload
from .__base import (Number, Shape, DataType, ArrayLike)

class Backend:
    """
    Defines the different computational backends where computations are executed

    Members:

      Default : It would resolve to the first available backend out of CUDA, CPU and OpenCL.

      CPU : Uses CPU multicore capabilities.

      CUDA : Executes algorithms in CUDA devices.

      OpenCL : Executes algorithms in OpenCL devices.
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
    CPU: shapelets.compute._pygauss.Backend # value = <Backend.CPU: 1>
    CUDA: shapelets.compute._pygauss.Backend # value = <Backend.CUDA: 2>
    Default: shapelets.compute._pygauss.Backend # value = <Backend.Default: 0>
    OpenCL: shapelets.compute._pygauss.Backend # value = <Backend.OpenCL: 4>
    __members__: dict # value = {'Default': <Backend.Default: 0>, 'CPU': <Backend.CPU: 1>, 'CUDA': <Backend.CUDA: 2>, 'OpenCL': <Backend.OpenCL: 4>}

class BorderType:
    """
    Border type

    Members:

      Zero : Values are 0

      Symmetric : Values are symmetric over the edge.

      ClampEdge : Values are clamped to the edge.

      Periodic : Values are mapped to range of the dimension in cyclic fashion.
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
    ClampEdge: shapelets.compute._pygauss.BorderType # value = <BorderType.ClampEdge: 2>
    Periodic: shapelets.compute._pygauss.BorderType # value = <BorderType.Periodic: 3>
    Symmetric: shapelets.compute._pygauss.BorderType # value = <BorderType.Symmetric: 1>
    Zero: shapelets.compute._pygauss.BorderType # value = <BorderType.Zero: 0>
    __members__: dict # value = {'Zero': <BorderType.Zero: 0>, 'Symmetric': <BorderType.Symmetric: 1>, 'ClampEdge': <BorderType.ClampEdge: 2>, 'Periodic': <BorderType.Periodic: 3>}

class ConvDomain:
    """
    Convolution Domain

    Members:

      Auto : Automatically picks the right convolution algorithm

      Frequency : Perform convolution in frequency domain

      Spatial : Perform convolution in spatial domain
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
    Auto: shapelets.compute._pygauss.ConvDomain # value = <ConvDomain.Auto: 0>
    Frequency: shapelets.compute._pygauss.ConvDomain # value = <ConvDomain.Frequency: 2>
    Spatial: shapelets.compute._pygauss.ConvDomain # value = <ConvDomain.Spatial: 1>
    __members__: dict # value = {'Auto': <ConvDomain.Auto: 0>, 'Frequency': <ConvDomain.Frequency: 2>, 'Spatial': <ConvDomain.Spatial: 1>}

class ConvMode:
    """
    Convolution Mode

    Members:

      Default : Output of the convolution is the same size as input

      Expand : Output of the convolution is signal_len + filter_len - 1
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
    Default: shapelets.compute._pygauss.ConvMode # value = <ConvMode.Default: 0>
    Expand: shapelets.compute._pygauss.ConvMode # value = <ConvMode.Expand: 1>
    __members__: dict # value = {'Default': <ConvMode.Default: 0>, 'Expand': <ConvMode.Expand: 1>}

class DeviceInfo:
    """
            Data class describing a device where computations are run within the active backend.

            A computational backend may expose more than one device.  Use ::func::`~shapelets.get_devices` to
            get a complete list of devices found on the active backend.
            
    """
    def __repr__(self) -> str: ...
    @property
    def compute(self) -> str:
        """
        Compute capabilities of the device within the platform

        :type: str
        """
    @property
    def id(self) -> int:
        """
        Id for this device, which is unique within the backend.

        :type: int
        """
    @property
    def isDoubleAvailable(self) -> bool:
        """
        Returns true if Float64 is supported.

        :type: bool
        """
    @property
    def isHalfAvailable(self) -> bool:
        """
        Returns true if Float16 is supported.

        :type: bool
        """
    @property
    def name(self) -> str:
        """
        Descriptive device name as provided by the drivers in this system.

        :type: str
        """
    @property
    def platform(self) -> str:
        """
        Platform information associated to the backend and device.

        :type: str
        """

class DeviceMemory:
    """
            Describes how much memory is currently in used on a particular device.

            Use ::func::`~shapelets.get_device_memory` to report and populate memory usage. The function
            ::func::`~shapelets.device_gc` will force a synchronization and removal of temporal arrays.
            
    """
    def __repr__(self) -> str: ...
    @property
    def buffers(self) -> int:
        """
        Number of distinct buffers in use.

        :type: int
        """
    @property
    def bytes(self) -> int:
        """
        Number of bytes used.

        :type: int
        """
    @property
    def locked_buffers(self) -> int:
        """
        Number of distinct buffers currently locked.

        :type: int
        """
    @property
    def locked_bytes(self) -> int:
        """
        Number of bytes currently locked.

        :type: int
        """

class MatrixProfile:
    def __init__(self, arg0: ShapeletsArray, arg1: ShapeletsArray) -> None: ...
    @property
    def index(self) -> ShapeletsArray:
        """
        :type: ShapeletsArray
        """
    @index.setter
    def index(self, arg0: ShapeletsArray) -> None:
        """
        TODO
        """
    @property
    def profile(self) -> ShapeletsArray:
        """
        :type: ShapeletsArray
        """
    @profile.setter
    def profile(self, arg0: ShapeletsArray) -> None:
        """
        TODO
        """

class MatrixProperties:
    """
    Matrix Properties

    Members:

      Default : Default

      Transposed : Data needs to be transposed

      ConjugatedTransposed : Data needs to be conjugated transposed

      Conjugated : Data needs to be conjugate

      Upper : Matrix is upper triangular

      Lower : Matrix is lower triangular

      UnitDiagonal : Matrix diagonal contains unitary values

      Symmetric : Matrix is symmetric

      PositiveDefinite : Matrix is positive definite

      Orthogonal : Matrix is orthogonal

      TriDiagonal : Matrix is tri-diagonal

      BlockDiagonal : Matrix is block diagonal
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

    BlockDiagonal: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Symmetric: 512>
    Conjugated: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Conjugated: 4>
    ConjugatedTransposed: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.ConjugatedTransposed: 2>
    Default: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Default: 0>
    Lower: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Lower: 64>
    Orthogonal: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Orthogonal: 2048>
    PositiveDefinite: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.PositiveDefinite: 1024>
    Symmetric: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Symmetric: 512>
    Transposed: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Transposed: 1>
    TriDiagonal: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.TriDiagonal: 4096>
    UnitDiagonal: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.UnitDiagonal: 128>
    Upper: shapelets.compute._pygauss.MatrixProperties # value = <MatrixProperties.Upper: 32>
    __members__: dict # value = {'Default': <MatrixProperties.Default: 0>, 'Transposed': <MatrixProperties.Transposed: 1>, 'ConjugatedTransposed': <MatrixProperties.ConjugatedTransposed: 2>, 'Conjugated': <MatrixProperties.Conjugated: 4>, 'Upper': <MatrixProperties.Upper: 32>, 'Lower': <MatrixProperties.Lower: 64>, 'UnitDiagonal': <MatrixProperties.UnitDiagonal: 128>, 'Symmetric': <MatrixProperties.Symmetric: 512>, 'PositiveDefinite': <MatrixProperties.PositiveDefinite: 1024>, 'Orthogonal': <MatrixProperties.Orthogonal: 2048>, 'TriDiagonal': <MatrixProperties.TriDiagonal: 4096>, 'BlockDiagonal': <MatrixProperties.Symmetric: 512>}

class NormType:
    """
    Norm Type

    Members:

      Vector1 : Treats the input as a vector and returns the sum of absolute values

      VectorInf : Treats the input as a vector and returns the max of absolute values

      Vector2 : Treats the input as a vector and returns euclidean norm

      VectorP : Treats the input as a vector and returns the p-norm

      Matrix : Return the max of column sums

      MatrixInf : Return the max of row sums

      Singular : Returns the max singular value). Currently NOT SUPPORTED

      LPQ : Returns Lpq-norm

      Euclid : Same as Vector2
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
    Euclid: shapelets.compute._pygauss.NormType # value = <NormType.Vector2: 2>
    LPQ: shapelets.compute._pygauss.NormType # value = <NormType.LPQ: 7>
    Matrix: shapelets.compute._pygauss.NormType # value = <NormType.Matrix: 4>
    MatrixInf: shapelets.compute._pygauss.NormType # value = <NormType.MatrixInf: 5>
    Singular: shapelets.compute._pygauss.NormType # value = <NormType.Singular: 6>
    Vector1: shapelets.compute._pygauss.NormType # value = <NormType.Vector1: 0>
    Vector2: shapelets.compute._pygauss.NormType # value = <NormType.Vector2: 2>
    VectorInf: shapelets.compute._pygauss.NormType # value = <NormType.VectorInf: 1>
    VectorP: shapelets.compute._pygauss.NormType # value = <NormType.VectorP: 3>
    __members__: dict # value = {'Vector1': <NormType.Vector1: 0>, 'VectorInf': <NormType.VectorInf: 1>, 'Vector2': <NormType.Vector2: 2>, 'VectorP': <NormType.VectorP: 3>, 'Matrix': <NormType.Matrix: 4>, 'MatrixInf': <NormType.MatrixInf: 5>, 'Singular': <NormType.Singular: 6>, 'LPQ': <NormType.LPQ: 7>, 'Euclid': <NormType.Vector2: 2>}

class ParallelFor:
    def __iter__(self) -> ParallelFor: ...
    def __next__(self) -> ParallelFor: ...

class ScanOp:
    """
    Scan binary operations

    Members:

      Add : 

      Mul : 

      Min : 

      Max : 
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
    Add: shapelets.compute._pygauss.ScanOp # value = <ScanOp.Add: 0>
    Max: shapelets.compute._pygauss.ScanOp # value = <ScanOp.Max: 3>
    Min: shapelets.compute._pygauss.ScanOp # value = <ScanOp.Min: 2>
    Mul: shapelets.compute._pygauss.ScanOp # value = <ScanOp.Mul: 1>
    __members__: dict # value = {'Add': <ScanOp.Add: 0>, 'Mul': <ScanOp.Mul: 1>, 'Min': <ScanOp.Min: 2>, 'Max': <ScanOp.Max: 3>}

class ScopedBatch:
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

class ShapeletsArray:
    def __add__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __and__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __copy__(self) -> ShapeletsArray: 
        """
        Shallow copy
        """
    def __deepcopy__(self, memo: object) -> ShapeletsArray: 
        """
        Deep copy
        """
    def __eq__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ge__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __getitem__(self, selector: object) -> ShapeletsArray: ...
    def __gt__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __iadd__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __iand__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ilshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __imod__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __imul__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ior__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ipow__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __irshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __isub__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __itruediv__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ixor__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __le__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __lshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __lt__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __matmul__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Matrix multiplication
        """
    def __mod__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __mul__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ne__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __neg__(self) -> ShapeletsArray: ...
    def __or__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __pow__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __radd__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rand__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __repr__(self) -> str: ...
    def __rlshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rmatmul__(self, other: ArrayLike) -> ShapeletsArray: 
        """
        Matrix multiplication
        """
    def __rmod__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rmul__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __ror__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rpow__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rrshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rshift__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rsub__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rtruediv__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __rxor__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __setitem__(self, selector: object, value: ArrayLike) -> ShapeletsArray: ...
    def __sub__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __truediv__(self, other: ArrayLike) -> ShapeletsArray: ...
    def __xor__(self, other: ArrayLike) -> ShapeletsArray: ...
    def astype(self, type: DataType) -> ShapeletsArray: 
        """
        converts the array into a new array with the specified type
        """
    def display(self, precision: int = 4, transpose: bool = True) -> None: ...
    def same_as(self, arr_like: ArrayLike, eps: float_ = 0.0001) -> bool: 
        """
        Performs a element wise comparison between the arrays and returns True if the two arrays are the same (same dimensions, same values).
        """
    @property
    def H(self) -> ShapeletsArray:
        """
        Get the conjugate-transpose of the current array

        :type: ShapeletsArray
        """
    @property
    def T(self) -> ShapeletsArray:
        """
        Get the transposed the array

        :type: ShapeletsArray
        """
    @property
    def backend(self) -> Backend:
        """
        :type: Backend
        """
    @property
    def dtype(self) -> DataType:
        """
        Returns the numpy dtype describing the type of elements held by this array

        :type: DataType
        """
    @property
    def is_column(self) -> bool:
        """
        :type: bool
        """
    @property
    def is_row(self) -> bool:
        """
        :type: bool
        """
    @property
    def is_single(self) -> bool:
        """
        :type: bool
        """
    @property
    def is_vector(self) -> bool:
        """
        :type: bool
        """
    @property
    def itemsize(self) -> int:
        """
        Returns the size in bytes of each individual item held by this array

        :type: int
        """
    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions.

        :type: int
        """
    @property
    def shape(self) -> Shape:
        """
        :type: Shape
        """
    @property
    def size(self) -> int:
        """
        Returns the total number of elements of this array.

        :type: int
        """
    __array_priority__ = 30
    __hash__ = None

