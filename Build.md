# Build Instructions



## Windows Pre-Requisites
Ensure you have:

- git 
- python 3.8
- cmake
- 7 zip
- Windows 10 SDK or Visual Studio with C++/CLI
- Optional components:
    - VCPKG, whose root can be found on environment VCPKG_ROOT. 
    - ArrayFire installation

### Chocolatey setup 

    choco install git cmake 7zip
    choco install python --version=3.8.0


## Instructions
1. Download the latest version of the repository by executing:

    git clone --recurse-submodules https://gitlab.com/shapelets/shapelets/python-client.git

2. Ensure all python dependencies are present in your environment: ```pip install -r requirements.txt```

3. Run ```setup.py```.  
If you planning to develop, use the ```develop``` option, that is, ```python setup.py develop```, which will set up the necessary entries in the pip library folders to point to your development environment.  Once setup with ```develop``` option, further builds and changes will be automatically picked up.

4. Run tests to ensure everything is running as expected by issuing ```pytest``` on the root folder.



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


export LD_PRELOAD=/workspaces/shapelets/modules/shapelets/compute/.libs/libmkl_def.so:/workspaces/shapelets/modules/shapelets/compute/.libs/libmkl_avx2.so:/workspaces/shapelets/modules/shapelets/compute/.libs/libmkl_core.so:/workspaces/shapelets/modules/shapelets/compute/.libs/libmkl_intel_lp64.so:/workspaces/shapelets/modules/shapelets/compute/.libs/libmkl_intel_thread.so:/workspaces/shapelets/modules/shapelets/compute/.libs/libiomp5.so