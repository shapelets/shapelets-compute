cmake_minimum_required(VERSION 3.4...3.18)

project(PYTHONBINDINGS)

# Find the ArrayFire package.
find_package(ArrayFire CONFIG REQUIRED)

# Logging support
find_package(spdlog REQUIRED)

# Find the pybind11 package
find_package(pybind11 REQUIRED)

set(PYTHONBINDINGS_BASE_DIR "${PROJECT_SOURCE_DIR}")
set(PYTHONBINDINGS_SRC "${PYTHONBINDINGS_BASE_DIR}/src")
set(PYTHONBINDINGS_INC "${PYTHONBINDINGS_BASE_DIR}/include")

# Files to add to compilation
file(GLOB_RECURSE PYTHONBINDINGS_CODE
        "${PYTHONBINDINGS_SRC}/*.cpp"
        "${PYTHONBINDINGS_INC}/*.h" )

# modules and files
pybind11_add_module(PYTHONBINDINGS ${PYTHONBINDINGS_CODE})

# ensure the name is properly et
# VERSION ${VERSION_SHORT} cannot be applied since the output is a dynamic library.
set_target_properties(PYTHONBINDINGS
    PROPERTIES
        OUTPUT_NAME "shapelets")

## Declare include directories
target_include_directories(PYTHONBINDINGS
    PRIVATE
        ${ArrayFire_INCLUDE_DIRS}
        ${PYTHONBINDINGS_INC})

# Add compile definitions
target_compile_definitions(PYTHONBINDINGS PRIVATE AF_UNIFIED)

TARGET_LINK_LIBRARIES(PYTHONBINDINGS
        PRIVATE ArrayFire::af
        spdlog::spdlog
        ${CMAKE_DL_LIBS})