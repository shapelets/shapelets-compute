# Building Shapelets Solo

> **First and Foremost**: If you just have checkout the project, ensure you have initialized all submodules by executing: `git submodule update --init --recursive`.

Vcpkg should be considered as the main mechanism out of which dependencies are resolved.  However, Vcpkg maintainers have a curious attitude towards header only libraries, usually fixing build parameters (spdlog) or making them dependant of other libraries (pybind11).

This project goes miles to ensure all dependencies can be resolved automatically and, at the same time, provide configuration settings that allows you to reuse existing infrastructure in your machine without having to download the world on each build.

When building this library, always have in mind which python version are you compiling for.  If necessary create either virtual environments, or simply use pyenv, to set for a particular python version.  Only versions 3.7 and above are relevant for this project; version 2.7 hasn't even tried.


## Dependencies

### [pybind11](https://github.com/pybind/pybind11)
This library provides the binding semantics for python.  It will be imported through a submodule and instantiated in `external\pybind11` folder.  Git is configured to track down the `stable` branch.

`pybind11` has a dependency to python libraries, which should be found automatically out of your current path, pyenv or virtualenv settings.  This is the primary reason for not including this library from vcpkg, since the build in that package manager is usually tied to the latest version of python, making the process of compilation against arbitrary python versions almost impossible.

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

### [ArrayFire](https://github.com/arrayfire/arrayfire)
`ArrayFire` is the principal library for vectorized computations.  It is really complex to build, so it is brought in binary format our of their _cdn_.

The default behaviour is to automatically download a copy of `ArrayFire`, regardless if a system wide installation is already present.  You can control this behaviour by using a cmake flag, `ArrayFire_Local`; when set to `OFF`, it will try to use the pre-installed version in your system.  The default behaviour is `ON`, that is, to ignore the system wide installation and to proceed with a controlled download (and version) which will expand in `external\arrayfire`.

The download process is executed by a python file located in `cmake\setup_af.py`; it is driven by a configuration file, `cmake\setup_af.json`, where you can amend the version more adequate to your build.  Usually, cmake will automatically trigger the download but it is possible to execute it in advance by running `cmake\setup_af.py`.  

To speed-up building tasks, and considering how huge these downloads are, packed distributions will be kept in `./.downloads/arrayfire/<<os>>`.  Therefore, you can delete / clean the external folder and it will be recreated for the cached file.

### [vcpkg](https://github.com/Microsoft/vcpkg)
For any other library that doesn't require any special treatment, `vcpkg` is used, using `vcpkg.json` file in the root project folder to document the libraries and versions required for the build.  

The build process will automate the download, set up and integration of vcpkg if no suitable installation is found.  The way the discovery works is quite straight forward and simply checks for the presence of an environment variable **`VCPKG_ROOT`**, which should be pointing to the main directory of your `vcpkg` installation. 

When `VCPKG_ROOT` is not defined, the build process will download and setup a copy of vcpkg automatically; the destination folder will be `external\vcpkg`.  

Please note the main `CMakeLists.txt` file will:

* set `VCPKG_FEATURE_FLAGS` to `manifest,versions`
* use `VCPKG_DEFAULT_TRIPLET` from the environment.  If not set, it will set `VCPKG_TARGET_ARCHITECTURE` to "x64" and `VCPKG_LIBRARY_LINKAGE` to "static".
* use `VCPKG_ROOT` from the environment; if not present, it will download `vcpkg`.


## First Steps -- Development version
1. Ensure you are happy with your environment settings
2. Make sure you have installed python requirements.
3. Run `./setup.py develop` or `python setup.py develop` to create a local installation working directly over the existing source code.
4. Optionally, run the tests to ensure everything goes fine.
5. Happy hacking!







