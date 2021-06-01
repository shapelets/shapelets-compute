# Building Shapelets Solo

> **First and Foremost**: If you just have checkout the project, ensure you have initialized all submodules by executing: ```git submodule update --init --recursive```.

This project tries to ensure all dependencies can be resolved automatically and, at the same time, provide configuration settings that allows you to reuse existing infrastructure in your machine.

When building this library, always have in mind which python version are you compiling for.  If necessary create either virtual environments, or simply use pyenv, to set for a particular python version.  Only versions 3.7 and above are supported.

## Windows Pre-Requisites
When building on windows, it is required the following components:

- `7 zip`: Required to unpack ArrayFire run-time distribution (if you don't have it installed already, the building process will download a copy and install it locally to this project).  
- `Windows 10 SDK` or `Visual Studio` with C++/CLI building tools.  
- CMake

If your environment has [Chocolatey](https://chocolatey.org/) setup, these dependencies can be installed quickly with the following command:

``` 
choco install git cmake 7zip
```

## Dependencies

These dependencies are resolved automatically at compile time; however, your environment may have already [VCPKG](https://github.com/microsoft/vcpkg) and [ArrayFire](https://arrayfire.com/).  If that is the case, read through to get some pointers that may speed up the building process by reusing your existing installation.

### ArrayFire [(Github)](https://github.com/arrayfire/arrayfire)
`ArrayFire` is the principal library for vectorized computations.  It is really complex to build, so it is brought in binary format out of their _cdn_.

The default behaviour is to automatically download a copy of `ArrayFire`, regardless if a system wide installation is already present.  You can control this behaviour by using a cmake flag, `ArrayFire_Local`; when set to `OFF`, it will try to use the pre-installed version in your system.  The default behaviour is `ON`, that is, to ignore the system wide installation and to proceed with a controlled download (and version) which will expand in `external\arrayfire`.

The download process is executed by a python file located in `cmake\setup_af.py`; it is driven by a configuration file, `cmake\setup_af.json`.  Normally, CMake will automatically trigger the download but it is possible to execute it in advance by running `cmake\setup_af.py`.  

To speed-up building tasks, and considering how huge these downloads are, packed distributions will be kept in `./.downloads/arrayfire/<<os>>`.  Therefore, you can delete / clean the external folder and it will be recreated for the cached file.

### [vcpkg](https://github.com/Microsoft/vcpkg)
For any other library that doesn't require any special treatment, `vcpkg` is used, using `vcpkg.json` file in the root project folder to outline the libraries and versions required for the build.  

The build process will automate the download, set up and integration of `vcpkg` if no suitable installation is found.  

To reuse an existing installation, ensure the presence of an environment variable `VCPKG_ROOT` or `VCPKG_INSTALLATION_ROOT`, which should be pointing to the main directory of your `vcpkg` installation. 

When `VCPKG_ROOT` is not defined, the build process will download and setup a copy of vcpkg automatically; the destination folder will be `external\vcpkg`.  

Please note the main `CMakeLists.txt` file will:

* set `VCPKG_FEATURE_FLAGS` to `manifest,versions`
* use `VCPKG_DEFAULT_TRIPLET` from the environment.  If not set, it will set `VCPKG_TARGET_ARCHITECTURE` to "x64" and `VCPKG_LIBRARY_LINKAGE` to "static".


### [pybind11](https://github.com/pybind/pybind11)
`shapelets-compute` uses `pybind11` to create the bindings for python.  `pybind11` will be imported through a git submodule and instantiated in `external\pybind11` folder.  Git is configured to track down the `stable` branch.

`pybind11` has a dependency to python libraries, which should be found automatically out of your current path, pyenv or virtualenv settings.  This is the primary reason for not including this library from `vcpkg`, since the build in that package manager is usually tied to the latest version of python, making the process of compilation against arbitrary python versions almost impossible.

`pybind11` is added to the project through the main `CMakeLists.txt` file, simply by adding it as a subdirectory, before any of the other modules.  `pygauss` module would, in turn, make use of `pybind11_add_module` function to hook up with the library.


### [spdlog](https://github.com/gabime/spdlog)
This library provides logging capabilities to CXX environments.  Similarly to `pybind11`, this is a headers only library but, `vcpkg`, breaks this assumption by making it dependant of `fmt` library.  `spdlog` has its own implementation for `fmt` so nothing is really lost when used as headers only.

`spdlog` is brought as a submodule in the folder `external\spdlog` and it is set to track v1.x branch.

When defining your own formatters, please make sure you define the imports in the following order:

```c++
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
```

and not the other way around.  `spdlog/fmt/ostr.h` will import the usual constructs you would expect from the `fmt` library.

## First Steps -- Development Build
1. Ensure you are happy with your environment settings.
2. Make sure you have installed python requirements: ```pip install -r requirements.txt``` and ```pip install -r requirements-test.txt```
3. Run ```./setup.py develop``` or ```python setup.py develop``` to create a local installation working directly over the existing source code.
4. Optionally, run the tests to ensure everything goes fine.
5. Run tests to ensure everything is running as expected by issuing ```pytest``` on the root folder.
6. Happy hacking!

## Creating a distribution
Install tox (```pip install tox```) to run an automated build and test cycle.  ```tox.ini``` is configured at root folder and it will build and test a distribution for Python 3.7, 3.8 and 3.9.  If you are using ```pyenv``` to controll your python environment, do install ```tox-pyenv```.

Binary wheels built after executing ```tox``` will be found in the ```dist``` folder.  

The versioned name of the wheel is controlled through ```versioneer``` and, if the git status is not clean, your wheels will be flagged as dirty.  The actual version number will be extracted from the latest git tag, which are expected in this format _**v**Major.Minor.Build_ (example: v0.2.1)

