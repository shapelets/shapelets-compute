.. _whatissc:

What is Shapelets-Compute?
##########################

Within Shapelets_ software stack, **Shapelets-Compute** is a recollection of algorithms 
and computational facilities to work efficiently with time series represented as dense arrays.  

The server side version of Shapelets, which deals with complexities like storage, indexing, ingestion, etc..., 
exposes a super set of the API exposed by Shapelets-Compute, but with a very different aim:  In Shapelets, 
algorithms are transformed into a `DAG (directed acyclic graph) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_, 
describing a distributed computation; the leaf nodes of such graphs are operations physically implemented by worker nodes, 
who uses Shapelets-Compute to execute operations and instructions over sequences of data.

Therefore, **Shapelets-Compute** offers you the same computational semantics as Shapelets server side, with the 
benefit of running it as open source library, directly in your machine without complex deployments nor the 
need to adquire server resources. 

Shapelets-Compute should help you during your investigation phase, where versatility and computational 
efficiency allows to try new analytical pipelines as fast as possible.

Shapelets-Compute API
---------------------

When defining an API suitable to data scientist and data engineers, we wanted to reuse some of the existing 
semantics of libraries like NumPy_, to reduce as much as possible the learning curve; 
at the same time, we set ourselves an objective to be as efficient as possible in terms of execution time, 
maximizing the facilities offered by from multicore hosts, OpenCL and CUDA devices.  

Therefore, whilst examining Shapelets-Compute, you will see that:

* ArrayFire_ is used internally as the underlying driver for most of our algorithms; 
  this allows you to choose among different backends and devices to accelerate your computations without having 
  to change a single like of code.

* Shapelets-Compute goes at great length to offer a similar API to NumPy, creating a seamless experience when 
  coding new algorithms to run in either Shapelets-Compute or Shapelets server side.

* *ShapeletsArrays* are our internal abstraction to work with dense array and tensors.  The 
  compatibility with the rest the Python infrastructure and libraries is achieved by implementing 
  `Buffer Protocol`_ on top of ArrayFire_ abstractions.

Next Steps
----------

1. Follow the installation guide
2. Play with some examples and tutorials
3. Read about array operations.




.. _ArrayFire: https://arrayfire.com/
.. _`Buffer Protocol`: https://docs.python.org/3/c-api/buffer.html
.. _NumPy: https://numpy.org/
.. _Shapelets: https://shapelets.io





