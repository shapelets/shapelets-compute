.. _backend:

Backend and Devices
===================

Three different backends are supported in Shapelets-Compute: `cpu`, `opencl` and `cuda`.  

Right after importing the library, the runtime is initialized and a backend is selected 
from the list of available backends, selecting the first backend found in this 
order: `cuda` -> `opencl` -> `cpu`.

The availability of a backend is determined by two factors: 

a) The backend runtime is available, task that can be easily achieve through the commandline tool 
   that ships with this package (see :ref:`user/installguide:Checking Your Installation` in the installation guide).
b) The operating system has the necessary drivers and libraries your environment.  

Leaving aside CUDA installations, `cpu` and `opencl` backends are usually available without requiring 
the installation of drivers and extra libraries.  If the machine doesn't have a powerful 
graphics card, one may consider setting up `opencl` CPU drivers.

When working with *Shapelets-Compute*, backend and device selection usually happens once.  Only in 
advanced scenarios where performance is a must, multiple backends and scenarios could be utilised 
simultaneously.  

The key take away when working with arrays is to keep in mind array's memory is associated with the 
backend and device active at the point of instantiation.  

Consider the following example:

>>> import shapelets.compute as sc 
>>> sc.get_devices()
[[0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘,
 [1] Intel(R)_HD_Graphics_630 (OpenCL - 1.2 - Apple) F64: ✘ - F16: ✘]
>>> sc.set_device(0)
>>> a = sc.array([1,2,3,4])
>>> sc.set_device(1)
>>> b = sc.array([5,6,7,8])

If we try to run an operation with those two arrays, the run time will complain almost 
immediately:

>>> a + b 
RuntimeError: Input Array not created on current device

Transferring memory between devices is an operation, or between device and host memory, is 
potentially an expensive operation and it should be done when it is required.  To avoid 
accidental transfers, the operation has to be done through the usage of an intermediary 
data holder, which cannot be a ShapeletsArray instance.  

For example, following with the previous code samples, we can use a :obj:`~memoryview` object
to create a copy from array `a` to main memory and then to transfer those contents to another 
device.  The allocated memory for the transfer will be automatically freed when the garbage 
collector kicks in.

>>> sc.array(a) + b # Fails by design in transferring data from device. 
RuntimeError: Input Array not created on current device
>>> sc.set_device(0)
>>> memview = memoryview(a) # Move Device 0 to host memory
>>> sc.set_device(1)
>>> sc.array(memview) + b   # Host memory to device 1
[4 1 1 1]
         6 
         8 
        10 
        12 

Device Capabilities
-------------------
To inspect the capabilities of a particular device, the function :obj:`~shapelets.compute.get_device` 
returns information about the active device.  

>>> sc.get_device()
AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘

Besides the descriptive texts documenting the name, platform and compute version, two flags to watch for 
are the ones informing if the device supports ``float64`` and ``float16``.  Trying to instantiate an array with 
an unsupported type will result in failure:

>>> sc.get_device()
[0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘
>>> sc.array([1,2,3], dtype=float16)
RuntimeError: ArrayFire Exception (Half precision floats not supported for this device:403):
In function void opencl::(anonymous namespace)::verifyTypeSupport()
In file src/backend/opencl/Array.cpp:73
Half precision not supported

Evaluation
----------
Operations over arrays are inherently asynchronous; the :obj:`~shapelets.compute.ShapeletsArray.eval` method forces 
the execution to wait until a particular result is fully computed.  For a complete synchronous (blocking) wait until 
all computations are completed, use the function :obj:`~shapelets.compute.sync`.  

The following example highlights the effect of evaluation and synchronization:

.. ipython:: 

   In [299]: import shapelets.compute as sc

   In [300]: a = sc.random.randn((1000,1000)) # create a 1M element matrix.

   In [301]: %timeit b = a * a
   2.74 µs ± 45.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

   In [302]: %timeit b = a * a; b.eval()
   51.2 µs ± 691 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

   In [303]: %timeit b = a * a; b.eval(); sc.sync()
   639 µs ± 20.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

Please note that before any operation that implies transfer from device to host memory (constructing an NumPy array, 
plotting, etc...) an :obj:`~shapelets.compute.ShapeletsArray.eval` is called transparently to ensure any pending
operation is completed.

There are scenarios, mainly loops, where it may be interesting to control or force the evaluation of a particular 
array to ensure temporal / support arrays are garbage collected before the next iteration (use the function 
:obj:`~shapelets.compute.device_gc()` to trigger garbage collection).


