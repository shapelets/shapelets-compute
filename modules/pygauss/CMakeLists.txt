cmake_minimum_required(VERSION 3.4...3.18)

project(PyGauss)

# Find the ArrayFire package.
find_package(ArrayFire CONFIG REQUIRED)

# Logging support
find_package(spdlog REQUIRED)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

set(PYGAUSS_BASE_DIR "${PROJECT_SOURCE_DIR}")
set(PYGAUSS_SRC "${PYGAUSS_BASE_DIR}/src")
set(PYGAUSS_INC "${PYGAUSS_BASE_DIR}/include")

# Files to add to compilation
file(GLOB_RECURSE PYGAUSS_CODE
        "${PYGAUSS_SRC}/*.cpp"
        "${PYGAUSS_INC}/*.h" )

# modules and files
pybind11_add_module(PyGauss ${PYGAUSS_CODE})

# ensure the name is properly et
# VERSION ${VERSION_SHORT} cannot be applied since the output is a dynamic library.
set_target_properties(PyGauss
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        OUTPUT_NAME "pygauss")

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
