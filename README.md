
mac
  - https://pypi.org/project/delocate/

```shell
DYLD_LIBRARY_PATH=/Users/justo.ruiz/Development/shapelets/solo_comprobacion/external/arrayfire/lib:/opt/arrayfire/lib/libaf:/Users/justo.ruiz/Development/shapelets/solo_comprobacion/temp/Developer/NVIDIA/CUDA-10.1/lib:$DYLD_LIBRARY_PATH delocate-listdeps --all dist/shapelets-0.1.1+5.g8dfcc89.dirty-cp37-cp37m-macosx_10_14_x86_64.whl 

DYLD_LIBRARY_PATH=/Users/justo.ruiz/Development/shapelets/solo_comprobacion/external/arrayfire/lib:/opt/arrayfire/lib/libaf:/Users/justo.ruiz/Development/shapelets/solo_comprobacion/temp/Developer/NVIDIA/CUDA-10.1/lib:$DYLD_LIBRARY_PATH delocate-wheel -w ./dist/fixed_wheels ./dist/shapelets-0.1.1+5.g8dfcc89.dirty-cp37-cp37m-macosx_10_14_x86_64.whl 
```

windows
  - https://vinayak.io/2020/10/22/day-52-bundling-dlls-with-windows-wheels-the-dll-mangling-way/
  - https://vinayak.io/2020/10/21/day-51-bundling-dlls-with-windows-wheels-the-package-data-way/

linux
  - https://github.com/pypa/auditwheel
  - https://github.com/njsmith/machomachomangler


Shapelets provides:

- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities
- and much more

Besides its obvious scientific uses, NumPy can also be used as an efficient
multi-dimensional container of generic data. Arbitrary data-types can be
defined. This allows NumPy to seamlessly and speedily integrate with a wide
variety of databases.

All Shapelets wheels distributed on PyPI are BSD licensed.



check out git clone --recurse-submodules
si se te olvida
git submodule update --init --recursive.

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

pybind11-stubgen shapelets.compute --ignore-invalid all  -o ./out 



# PYBind11
Why is it external? Because vcpkg automatically gets the latest pyton environment.  This should be a headers only library whose 
rt dependencies should be determined through virtual envs.

#Spdlog
vcpkg does a very odd thing here: it forces the compilation with the external fmt library, rather than using the header only 
version that ships with spdlog.  there is no way of fixing this since it "compiles" at instalation fixing the external nature 
of fmt (which then becomes dynamically linked regardless of preferences in windows.)
