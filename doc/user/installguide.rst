.. _installguide:

==================
Installation Guide
==================
To install a binary distribution, the only requirement is to have a working (recent, 
3.7 and onwards) version of Python in your system.  

+++
PIP 
+++
The installation process is a two step process.  The first step is to install 
**Shapelets-Compute** as a conventional package::

    pip install shapelets-compute

Due to the large size of runtime depedencies (MKL, CUDA, OpenCL), there is a second 
step in the installation process; PyPI repositories impose size restrictions on the 
packages they serve.

To install the required runtime libraries, one could proceed to download it directly 
from `ArrayFire's downloads`_ site.  If you already have a valid installation of 
ArrayFire in your environment, you validate that everything is working correctly by
following the steps outlined in the section :ref:`Checking Your Installation`.

Our recommended approach to install run time depedencies is to use the command line 
tool, `shapelets`, to set up the exact binaries Shapelets-Compute has been compiled 
with.  The process is extremelly simple::

    shapelets install <<backend>>

where `backend` is one of the following `cpu`, `opencl` and/or `cuda`.  Multiple 
backends are supported by simply installing them individually.  

.. note::

    When installing `opencl` or `cuda`, your environment should have the 
    required devide drivers.
    
.. admonition:: Writing permissions

    Installing runtime libraries using the command line tool will require 
    writing permissions in your file system.  A temporal folder will be created 
    (usually an operation that doesn't carry any permissions) and it will unzip 
    the files `.libs` folder; the exact folder depends on you environment but it 
    can easily be checked issuing a ```$ shapelets info``` command.


Checking Your Installation
--------------------------
To ensure your installation is working, use the commandline tool that ships with 
Shapelets-Compute.  Simply issue an ```info``` command, like the one shown::

    $ shapelets info 

    Shapelets version : 0.2.2 [37]
    Platform Libraries: <<path to the binary runtime folder>>
    Default backend   : opencl
    Default device    : 0

    Available backend and devices
    cpu     [0] Intel (CPU - 0.0 - AppleClang 10.0.0.10001044) F64: ✓ - F16: ✓
    opencl  [0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘
    opencl  [1] Intel(R)_HD_Graphics_630 (OpenCL - 1.2 - Apple) F64: ✘ - F16: ✘

The ```info``` command will report the version of the library (in square brackets, 
ArrayFire's version); *Platform Libraries* indicates the location where the runtime 
binaries are located.  

The rest of the information documents the backends (cpu, opencl or cuda) that will be 
used by default and the total list of backends and devices found in your system.

To test and evaluate the performance of each one of those environments, the command line 
tool offers the option ```bench``` that executes and reports the performance in typical 
computation workflows like matrix multiplication (```blas```) or Fast Fourier Transform 
(```fft```).  For example::

    $ shapelets bench opencl -t float32 blas 
    Running benchmark blas for opencl[0] using float32
    [0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘

     512 | 149.805742 ████
    1024 | 780.899936 █████████████████████▎
    1536 | 828.657096 ██████████████████████▋
    2048 | 873.775113 ███████████████████████▊
    2560 | 891.639470 ████████████████████████▎
    3072 | 915.047254 █████████████████████████
    3584 | 909.533975 ████████████████████████▊
    4096 | 860.105441 ███████████████████████▍


+++++
Conda
+++++
Conda installation is, currently, work in progress

++++++++++++++++++++++++
Installation from source
++++++++++++++++++++++++

**First and Foremost**: If you just have check out the project,
ensure you have initialized all submodules by executing:
``git submodule update --init --recursive``.

This project tries to ensure all dependencies can be resolved
automatically and, at the same time, provide configuration settings that
allow you to reuse existing infrastructure in your machine.

When building this library, always have in mind which python version are
you compiling for. If necessary create either virtual environments, or
simply use pyenv, to set for a particular python version. Only versions
3.7 and above are supported.

Windows Pre-Requisites
----------------------
When building on windows, it is required the following components:

- ``7 zip``: Required to unpack ArrayFire run-time distribution (if you
  don't have it installed already, the building process will download a
  copy and install it locally to this proyect).
- ``Windows 10 SDK`` or ``Visual Studio`` with C++/CLI building tools.
- CMake

If your environment has `Chocolatey`_, these dependencies can be
installed quickly with the following command::

    choco install cmake 7zip

Darwin Pre-Requisites
---------------------
If you are using a Mac, the only requirement is to ensure you have installed 
the *Command Line Tools* (which can be installed by issuing the following command 
in your terminal console: ```xcode-select —install```)

Linux Pre-Requisites
--------------------
Usual tools for development in Linux environments are the only requirement, more 
specifically, a C++ 17 compiler.


Dependencies
------------
These dependencies are resolved automatically at compile time; however,
your environment may have already `VCPKG`_ and `ArrayFire`_. If that is
the case, read through to get some pointers that may speed up the
building process by reusing your existing installation.

ArrayFire `(Github)`_
~~~~~~~~~~~~~~~~~~~~~
``ArrayFire`` is the principal library for vectorized computations. It
is really complex to build, so it is brought in binary format.

The default behaviour is to automatically download a copy of
``ArrayFire``, even if a system wide installation is already
present. You can control this behaviour by using a cmake flag,
``ArrayFire_Local``; when set to ``OFF``, it will try to use the
pre-installed version in your system. The default behaviour is ``ON``,
that is, to ignore the system wide installation and to proceed with a
controlled download (and version) which will expand in
``external\arrayfire``.

The download process is executed by a python file located in
``cmake\setup_af.py``; it is driven by a configuration file,
``cmake\setup_af.json``. CMake will automatically trigger the
download, but it is possible to execute it in advance by running
``cmake\setup_af.py``.

`vcpkg`_
~~~~~~~~
For any other library that doesn't require any special treatment,
``vcpkg`` is used, using ``vcpkg.json`` in the root project folder
to outline the libraries and versions required for the build.

The build process will automate the download, set up and integration of
``vcpkg`` if no suitable installation is found.

To reuse an existing installation, ensure the presence of an environment
variable ``VCPKG_ROOT`` or ``VCPKG_INSTALLATION_ROOT``, which should be
pointing to the main directory of your ``vcpkg`` installation.

When ``VCPKG_ROOT`` is not defined, the build process will download and
setup a copy of vcpkg automatically; the destination folder will be
``external\vcpkg``.

`pybind11`_
~~~~~~~~~~~
``shapelets-compute`` uses ``pybind11`` to create the bindings for
Python. ``pybind11`` will be imported through a git submodule and
instantiated in ``external\pybind11`` folder. Git is configured to track
the stable branch.

``pybind11`` has a dependency to python libraries, which should be found
automatically out of your current path, pyenv or virtualenv settings.

This is the primary reason for not including this library from
``vcpkg``, since the build in that package manager is usually tied to
the latest version of python, making the process of compilation against
arbitrary python versions almost impossible.

`spdlog`_
~~~~~~~~~
This library provides logging capabilities to CXX environments.
Similarly to ``pybind11``, this is a headers only library but,
``vcpkg``, breaks this assumption by making it dependant of ``fmt``
library. ``spdlog`` has its own implementation for ``fmt`` so nothing is
really lost when used as headers only.

``spdlog`` is brought as a submodule in the folder ``external\spdlog``
and it is set to track v1.x branch.


First Steps -- Development Build
--------------------------------
1. Ensure you are happy with your environment settings.
2. Make sure you have installed python requirements:
   ``pip install -r requirements.txt`` and
   ``pip install -r requirements-test.txt``
3. Run ``./setup.py develop`` or ``python setup.py develop`` to create a
   local installation working directly over the existing source code.
4. Optionally, run the tests to ensure everything goes fine.
5. Run tests to ensure everything is running as expected by issuing
   ``pytest`` on the root folder.
6. Happy hacking!

Creating a distribution
-----------------------
Install tox (``pip install tox``) to run an automated build and test
cycle. ``tox.ini`` is configured at root folder and it will build and
test a distribution for Python 3.7, 3.8 and 3.9. If you are using
``pyenv`` to controll your python environment, do install ``tox-pyenv``.

Binary wheels built after executing ``tox`` will be found in the
``dist`` folder.

The versioned name of the wheel is controlled through ``versioneer``
and, if the git status is not clean, your wheels will be flagged as
dirty. The actual version number will be extracted from the latest git
tag, which are expected in this format **v**\ *\ Major.Minor.Build*
(example: v0.2.1)

.. _Chocolatey: https://chocolatey.org/
.. _vcpkg: https://github.com/microsoft/vcpkg
.. _ArrayFire: https://arrayfire.com/
.. _(Github): https://github.com/arrayfire/arrayfire
.. _`ArrayFire's downloads`: https://arrayfire.com/download/
