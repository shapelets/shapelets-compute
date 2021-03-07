cmake_minimum_required(VERSION 3.4...3.18)

project(PyGauss)

# Find the ArrayFire package.
find_package(ArrayFire CONFIG REQUIRED)

# Logging support
find_package(spdlog REQUIRED)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

# Find arrow
find_package(arrow CONFIG REQUIRED)

set(PYGAUSS_BASE_DIR "${PROJECT_SOURCE_DIR}")
set(PYGAUSS_SRC "${PYGAUSS_BASE_DIR}/src")
set(PYGAUSS_INC "${PYGAUSS_BASE_DIR}/include")

# Files to add to compilation
file(GLOB_RECURSE PYGAUSS_CODE
        "${PYGAUSS_SRC}/*.cpp"
        "${PYGAUSS_INC}/*.h")

# modules and files
pybind11_add_module(PyGauss ${PYGAUSS_CODE})

# ensure the name is properly et
# VERSION ${VERSION_SHORT} cannot be applied since the output is a dynamic library.
set_target_properties(PyGauss
        PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        OUTPUT_NAME "_pygauss")

## Declare include directories
target_include_directories(PyGauss
        PRIVATE
        ${PYGAUSS_INC})

# Add compile definitions
target_compile_definitions(PyGauss PRIVATE AF_UNIFIED)

target_link_libraries(PyGauss
        PRIVATE
        ArrayFire::af
        gauss
        spdlog::spdlog
        ${CMAKE_DL_LIBS})


#########
## STUBS
##
## These libraries are built with the sole purpose to include external dependencies
## in the final assembly.  They don't implement any functionality of any use, besides
## capturing run time dependencies with array fire.
#########

# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

## cpu
# pybind11_add_module(stubcpu "${PYGAUSS_BASE_DIR}/stub/stub.cpp")
# target_compile_definitions(stubcpu PRIVATE STUB_NAME=stubcpu)
# target_link_libraries(stubcpu
#         PRIVATE ${CMAKE_DL_LIBS}
#         PUBLIC ArrayFire::afcpu)
# set_target_properties(stubcpu
#         PROPERTIES
#         OUTPUT_NAME "stubcpu")

## cuda
# pybind11_add_module(stubcuda "${PYGAUSS_BASE_DIR}/stub/stub.cpp")
# target_compile_definitions(stubcuda PRIVATE STUB_NAME=stubcuda)
# target_link_libraries(stubcuda
#         PRIVATE ${CMAKE_DL_LIBS}
#         PUBLIC ArrayFire::afcuda)
# set_target_properties(stubcuda
#         PROPERTIES
#         OUTPUT_NAME "stubcuda")

## opencl
# pybind11_add_module(stubcl "${PYGAUSS_BASE_DIR}/stub/stub.cpp")
# target_compile_definitions(stubcl PRIVATE STUB_NAME=stubcl)
# target_link_libraries(stubcl
#         PRIVATE ${CMAKE_DL_LIBS}
#         PUBLIC ArrayFire::afopencl)
# set_target_properties(stubcl
#         PROPERTIES
#         OUTPUT_NAME "stubcl")
