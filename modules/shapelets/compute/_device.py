from __future__ import annotations
from typing import Optional, Sequence, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from ._pygauss import (DeviceInfo, DeviceMemory)
from . import _pygauss

# Different backend types
Backend = Literal['cpu', 'opencl', 'cuda', 'default']

class DeviceInfo:
    """
    Describes the current device and informs if 64 or 16 bit floating point structures are supported.
    """
    def __repr__(self) -> str: ...

    @property
    def compute(self) -> str: 
        """
        Name of the computational platform
        """

    @property
    def id(self) -> int: 
        """
        Unique identifier of the device (within the context of the active backend)
        """

    @property
    def isDoubleAvailable(self) -> bool: 
        """
        Informs if 64 bit floating point arithmetic is supported
        """

    @property
    def isHalfAvailable(self) -> bool: 
        """
        Informs if 16 bit floating point arithmetic is supported
        """

    @property
    def name(self) -> str: 
        """
        Name of the device
        """

    @property
    def platform(self) -> str: 
        """
        Name of the computational platform
        """

class DeviceMemory:
    """
    Provides information about the memory sizes and status on a particular device
    """

    def __repr__(self) -> str: ...

    @property
    def buffers(self) -> int: 
        """
        Number of buffers currently allocated on a particular device
        """

    @property
    def bytes(self) -> int:
        """
        Number of bytes currently allocated on a device
        """

    @property
    def locked_buffers(self) -> int: 
        """
        Out of all buffers allocated, this property informs on how many of those are pinned or locked.
        """

    @property
    def locked_bytes(self) -> int:
        """
        Out of all the bytes allocated on a device, this property reports how many are currently being
        used, locked or pinned.
        """

def __to_pygauss_backend(backend: Backend):
    if backend == 'cpu':
        return _pygauss.Backend.CPU
    elif backend == 'opencl':
        return _pygauss.Backend.OpenCL
    elif backend == 'cuda':
        return _pygauss.Backend.CUDA 
    elif backend == 'default':
        return _pygauss.Backend.Default
    else:
        raise ValueError("Unkown backend type")

def __backend_as_literal(backend: _pygauss.Backend) -> Backend:
    if backend == _pygauss.Backend.CPU:
        return 'cpu'
    elif backend == _pygauss.Backend.OpenCL:
        return 'opencl'
    elif backend == _pygauss.Backend.CUDA:
        return 'cuda'
    return 'default'


def device_gc() -> None:
    """
    Forces a garbage collection operation on the current device

    Notes
    -----
    Usually, memory pressure and deallocation is handled automatically.  However, if an 
    operation or algorithm allocates memory intensively, one could force a deallocation 
    of memory buffers by invoking this function explicetly.

    To check the effect of the operation, use :obj:`~shapelets.compute.get_device_memory` and 
    inspect the properties :obj:`~shapelets.compute.get_device_memory.bytes` and 
    :obj:`~shapelets.compute.get_device_memory.locked_bytes`

    See Also
    --------
    get_device
        Returns the current device.
    
    get_backend
        Returns the current backend.
    
    get_device_memory
        Returns information about the current memory pressure on a particular device.

    Examples
    --------
    >>> import shapelets.compute as sc 
    >>> sc.get_device_memory() 
    bytes: 640, buffers: 1, locked_bytes: 0, locked_buffers: 0
    >>> sc.device_gc()
    >>> sc.get_device_memory() 
    bytes: 0, buffers: 0, locked_bytes: 0, locked_buffers: 0
    """
    return _pygauss.device_gc()

def get_available_backends() -> Sequence[Backend]:
    """
    Returns a sequence of available backends in the current platform.
    """
    return [__backend_as_literal(b) for b in _pygauss.get_available_backends()]

def get_backend() -> Backend:
    """
    Returns the active backend.
    """
    return __backend_as_literal(_pygauss.get_backend())

def get_device_memory(dev: Optional[Union[int, DeviceInfo]] = None) -> DeviceMemory:
    """
    Gets memory information on a particular device

    Parameters
    ----------
    dev: Optional, defaults to None
        When unset, it returns memory information of the current device; to query memory 
        of a different device, use either the :obj:`~shapelets.compute.DeviceInfo.id` property
        on :obj:`~shapelets.compute.DeviceInfo` or the actual :obj:`~shapelets.compute.DeviceInfo` object
        itself.

    Returns
    -------
    DeviceMemory
        A new instance of a :obj:`~shapelets.compute.DeviceMemory` class

    Example
    -------
    To query the memory of the current device:

    >>> import shapelets.compute as sc
    >>> sc.get_device_memory()
    bytes: 0, buffers: 0, locked_bytes: 0, locked_buffers: 0

    To query all devices in the current backend:
    
    >>> [sc.get_device_memory(d) for d in  sc.get_devices()]
    [bytes: 0, buffers: 0, locked_bytes: 0, locked_buffers: 0, bytes: 0, buffers: 0, locked_bytes: 0, locked_buffers: 0]
    """
    return _pygauss.get_device_memory(dev)


def get_device() -> DeviceInfo: 
    """
    Returns the current device in use.
    """
    return _pygauss.get_device()


def get_devices() -> Sequence[DeviceInfo]: 
    """
    Returns all devices in the current backend.
    """
    return _pygauss.get_devices()

def has_backend(test: Backend) -> int: 
    """
    Checks if a particular backend is available.

    Examples
    --------

    >>> import shapelets.compute as sc
    >>> sc.has_backend('opencl')
    True
    >>> sc.has_backend('cuda')
    False

    """
    return _pygauss.has_backend(__to_pygauss_backend(test))

def set_backend(backend: Backend) -> None: 
    """
    Changes the current backend.  
    """
    return _pygauss.set_backend(__to_pygauss_backend(backend))

def set_device(dev: Union[int, DeviceInfo]) -> bool: 
    """
    Changes the active device
    """
    return _pygauss.set_device(dev)

def sync(dev: Optional[Union[int, DeviceInfo]] = None) -> None: 
    """
    Sychronously wait for all operations and data transfers to finish.
    """
    return _pygauss.sync(dev)

def enable_manual_eval(new_value: bool) -> None:
    """

    """
    return _pygauss.enable_manual_eval(new_value)

def eval(*args) -> None: 
    """

    """
    return _pygauss.eval(*args)

def is_manual_eval_enabled() -> bool: 
    """
    
    """
    return _pygauss.manual_eval_enabled()


__all__=[
    "get_backend", "set_backend", "has_backend", "get_available_backends",
    "get_devices", "get_device", "set_device", "device_gc",
    "sync", "get_device_memory",
    "enable_manual_eval", "eval", "is_manual_eval_enabled",
    "DeviceInfo","DeviceMemory","Backend"
]
