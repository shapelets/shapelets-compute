.. _shapeletsarray:

Shapelets Arrays
================ 

This guide introduces ShapeletsArray, a wrapper around `ArrayFire <https://arrayfire.org/>`_ arrays, aiming 
to produce a close mirror of NumPy main APIs and behaviour. 

This document assumes prior knowledge of NumPy.  If you are new to NumPy, do please have a look at their online 
`tutorials and guides <https://numpy.org/doc/stable/user/basics.html>`_.  

TL;DR
-----
A quick summary of key aspects of Shapelets arrays:

- Arrays are allocated in device's memory using column major ordering.
- Operations are executed asynchronously and, in most cases, operations are combined and optimized by 
  ArrayFire's kernel fusion technology.
- Arrays can be built from any expression that builds a NumPy array.  
- Passing array objects to other Python libraries is supported, allowing a seamless integration of this 
  library with the rest of Python libraries and tools.
- There are minor differences with NumPy in areas (direct element access, one dimensional vectors) but, 
  in most cases, there is a like to like behaviour between both libraries.


Creating Arrays
---------------
There are basically two ways of creating arrays in *Shapelets-Compute*:

* Native functions that create arrays (see :ref:`reference/routines.construction:Array Creation` section 
  in the reference guide).
* Direct interface with array like objects in Python (lists, tuples, NumPy arrays, Pandas,
  memoryview objects, etc...)

Python Objects as Arrays
~~~~~~~~~~~~~~~~~~~~~~~~
The function ``array`` is the main entry point to convert any array like structure 
to a ShapeletsArray instance.  For example:

>>> import shapelets.compute as sc 
>>> a = sc.array([1,2,3])
>>> b = sc.array([(1,2,3), [3,4j,5]])  # mixed types!

The type of the newly created array will be inferred from the type of the parameters, finding 
a common ground representation, able to accurately represent all the values uniformly.  Usually, 
if at least one number is complex, the resulting array will be complex; similarly, if a number 
is a float, the resulting array will be a floating point array.  

Interfacing 
~~~~~~~~~~~
Since Shapelets-Compute arrays implement **array and buffer protocols**, the compatibility with 
existing libraries is almost guaranteed.  For example, to convert from and to NumPy arrays 
is extremely simple:

>>> import shapelets.compute as sc 
>>> import numpy as np 
>>> x = np.array([1,2,3])
>>> a = sc.array(x)
>>> y = np.array(a)
>>> x == y 
array([ True,  True,  True])

Plotting arrays with your favourite charting library is equally possible, without intermediate 
transformations:

.. plot::
    :include-source:

    import shapelets.compute as sc 
    import matplotlib.pyplot as plt 
    x = sc.arange(0., 2.0 * np.pi, 0.01)
    a = sc.sin(2*x)
    b = sc.cos(3*x)
    plt.plot(a, b)
    plt.show()

Direct array creation
~~~~~~~~~~~~~~~~~~~~~
:ref:`reference/routines.construction:Array Creation` section in the reference guide contains 
a comprehensive list of all the functions available for constructing arrays.  The API tries 
to mimic the functionality provided by NumPy, aiming to reduce the time it takes to 
start working with Shapelets-Compute.  

In general, you should find most of the functions for creating arrays ported already but do 
please let us know if we have missed an important one!

>>> import shapelets.compute as sc 
>>> x = sc.arange(0.0, 1.0, 0.01)
>>> y = sc.linspace([0,0],[50,100], 20)
>>> z = sc.ones_like(y)
>>> I = sc.identity((3,3))

Arrays Types
------------
:ref:`reference/routines.construction:Array Creation` routines, among others, support 
a parameter to specify the nature of the data type.  This parameter, ``dtype``, accepts 
the same constructs as :obj:`numpy.dtype`:

>>> import shapelets.compute as sc 
>>> import numpy as np 
>>> a = sc.array([1,2,3], dtype=np.single)  # float32
>>> b = sc.array([1,2,3], dtype='f')        # same as before
>>> c = sc.array([1,2,3], dtype=np.float32) # same as before
>>> d = sc.array([1,2,3], dtype="float32")  # same as before

When querying an array for its type, the result value will report the precise type 
in number of bits:

>>> a.dtype
dtype('float32')

As a rule of thumb, all numerical types supported in Python and NumPy are supported in Shapelets-Compute with the 
exception of structured types, 128 bits types and half/double precision, which are device dependent. Finally, object and 
variable length types are not supported.

The following table outlines supported types and caveats:

.. csv-table:: 
    :header: "Type","Supported","Unambiguous Alias","Comments"
    :widths: 15,15,15,70

    :obj:`numpy.bool8`,      ✓  , :obj:`numpy.bool_` :obj:`bool`, "Represented as uint8, where True is any value distinct to zero"
    :obj:`numpy.int8`,       ✘  , , "All bytes are represented as uint8.  If you require byte arithmetic, use 16 bit representation"
    :obj:`numpy.int16`,      ✓  , , 
    :obj:`numpy.int32`,      ✓  , , 
    :obj:`numpy.int64`,      ✓  , , 
    :obj:`numpy.uint8`,      ✓  , , 
    :obj:`numpy.uint16`,     ✓  , , 
    :obj:`numpy.uint32`,     ✓  , , 
    :obj:`numpy.uint64`,     ✓  , , 
    :obj:`numpy.float16`,    ???, :obj:`numpy.half`, Device depdendent
    :obj:`numpy.float32`,    ✓  , :obj:`numpy.single`, 
    :obj:`numpy.float64`,    ✓  , :obj:`numpy.double`, 
    :obj:`numpy.float96`,    ✘, , 
    :obj:`numpy.float128`,   ✘, , 
    :obj:`numpy.complex64`,  ✓  , :obj:`numpy.csingle`, 
    :obj:`numpy.complex128`, ???, :obj:`numpy.cdouble`, Device depdendent
    :obj:`numpy.complex192`, ✘, , 
    :obj:`numpy.complex256`, ✘, , 

.. warning:: 

    `Platform dependent types <https://numpy.org/doc/stable/user/basics.types.html#array-types-and-conversions-between-types>`_ are
    supported but not recommended, as their size may change from server to server and it may lead to incompatibilities between 
    your code and computational devices.


Dimensionality and broadcasting
-------------------------------
One key difference between NumPy and Shapelets arrays is dimensionality.  Whilst NumPy doesn't cap the maximum number of 
dimensions, in Shapelets the limit is set to 4.  

For most applications, this hard limit should not cause any issues; however, the way dimensions are treated may 
raise more than one eyebrow ever so often.  Let's take the following example:

>>> import numpy as np 
>>> a = np.array([1,2,3])
>>> b = a.T 
>>> a + b 
array([2, 4, 6])

And now execute the same code in Shapelets:

>>> import numpy as np 
>>> a = np.array([1,2,3])
>>> b = a.T 
>>> a + b 
[3 3 1 1]
         2          3          4 
         3          4          5 
         4          5          6 

**Ups!!** What is going on? 

Shapelets array have strong dimensionality semantics, that is, all arrays have 4 dimensions defined; 
even when the array is unidimensional, there is a strong difference between a columnar vector (``a`` in the 
example) and a row vector (``b`` or ``a.T``).  

In NumPy, one dimensional arrays are more fluid and the operation transpose over a one dimensional array 
doesn't really create any differences between ``a`` and ``b`` in terms of dimensionality.  Hence, the plus 
operation ``a + b`` is executing an element-wise addition between to vectors of dimensionality ``(3,)``

However, the same operation in Shapelets is computing an element-wise operation over a column vector (3x1) 
and a row vector (1x3).  Since the geometry of the operation requires broadcasting, the result is computed 
as matrix of (3x3) where each vector is tiled to produce a common geometry.  

This behaviour can be reproduced in NumPy by introducing a ``new axis`` construct:

>>> import numpy as np 
>>> a = np.array([1,2,3])
>>> b = a[np.newaxis,:].T
>>> a + b 
array([[2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]])

Putting aside the geometrical differences of one dimensional arrays, the behaviour of NumPy and Shapelets arrays is the same; 
broadcasting rules are equivalent in both systems, whereby two dimensions are compatible when

1. they are equal, or
2. one of them is 1

in a system fixed by 4 dimensions, avoiding right to left ordering.  


Indexing
--------
Array indexing takes the same form and semantics as in NumPy, for example:

>>> import shapelets.compute as sc 
>>> a = sc.iota((4,4), dtype=np.int32) # 4x4 array 
>>> a[:,0] # first column         
[4 1 1 1]
         0 
         1 
         2 
         3 
>>> a[0,:] # first row 
[1 4 1 1]
         0          4          8         12
>>> a[0, ...] # ellipsis are permitted
[1 4 1 1]
         0          4          8         12 
>>> a[::2, ...] # every second row
[2 4 1 1]
         0          4          8         12 
         2          6         10         14 
>>> a[-1, ...] # last row 
[1 4 1 1]
         3          7         11         15
>>> a[::-1, ...] # reverse all rows
[4 4 1 1]
         3          7         11         15 
         2          6         10         14 
         1          5          9         13 
         0          4          8         12 

Slicing can be used in assignment operations:

>>> a[1, ...] = -1 # sets second row to -1
>>> a[..., 1] = -1 # sets second column to -1
>>> a  
[4 4 1 1]
         0         -1          8         12 
        -1         -1         -1         -1 
         2         -1         10         14 
         3         -1         11         15 

Using other arrays as indexing is possible; the resulting selection will be the cartesian product of 
the selectors.  For example:

>>> a = sc.iota((4,4), dtype=np.int32) # 4x4 array 
>>> ic = sc.array([1,2])
>>> ir = sc.array([1,2])
>>> a[ir, ic]
[2 2 1 1]
         5          9 
         6         10 
>>> a[ir, ic] = -1
>>> a
[4 4 1 1]
         0          4          8         12 
         1         -1         -1         13 
         2         -1         -1         14 
         3          7         11         15  
         
Indexing caveats
~~~~~~~~~~~~~~~~
There are a few noteworthy differences some scenarios, which are important to highlight:

- When an array has more than one effective dimension, indexing by a single value returns 
  different results as single value indexing is interpreted as element selection (in 
  column major ordering).  For example:

  >>> a = sc.iota((4,4), dtype=np.int32) 
  >>> b = np.array(a)
  >>> a[3] 
  [1 1 1 1]
        3   
  >>> b[3]
  array([ 3,  7, 11, 15], dtype=int32)
  
  To mirror NumPy semantics, simply use the ellipsis operator to signify a row is required:

  >>> a[3, ...]
  [1 4 1 1]
         3          7         11         15 

- Scalar access. Whilst it is perfectly valid to access an individual value by its index, its 
  usage is **highly discouraged**.  
  
  In Shapelets, arrays are allocated directly in the device memory and operations are 
  asynchronous; accessing elements by position requires synchronization and data transfers 
  from your device memory to the host memory.  

  When accessing elements by their indices, Shapelets returns an array instance with a single 
  element, rather than a scalar, precisely to avoid synchronization and data transfer issues.  

  >>> a[2,3]
  [1 1 1 1]
        14   
  >>> int(a[2,3])  
  14

  .. note:: 

    During interactive Python sessions, printing results to the screen forces transparently synchronization
    and full evaluation of the asynchronous pipeline associated with the result in order to print 
    to the screen the desired results.  
    

