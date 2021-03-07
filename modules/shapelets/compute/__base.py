
from ._pygauss import (
    Backend, BorderType, ConvDomain, ConvMode, DeviceInfo, DeviceMemory, MatrixProfile,
    MatrixProperties, NormType, ParallelFor, ScanOp, ScopedBatch, ShapeletsArray)

from typing import Optional, Tuple, Union, overload

Number = Union[float, int, complex]
Shape = Union[int, Tuple[int, ...]]
DataType = Union[str, object]
ArrayLike = Union[ShapeletsArray, ParallelFor, Number, object]

__ALL__ = [
    "Number",
    "Shape",
    "DataType",
    "ArrayLike",
    "Backend", 
    "BorderType", 
    "ConvDomain", 
    "ConvMode", 
    "DeviceInfo", 
    "DeviceMemory", 
    "MatrixProfile",
    "MatrixProperties", 
    "NormType", 
    "ParallelFor", 
    "ScanOp", 
    "ScopedBatch", 
    "ShapeletsArray"]    

