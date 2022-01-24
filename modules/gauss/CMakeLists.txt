# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.9.6)

project(Gauss)

#sanity check
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/scamp/CMakeLists.txt")
	message(FATAL_ERROR "git submodule was not initialized.")
endif()

find_package(ArrayFire CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

find_package(CUDA )
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 14)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    
	enable_language(CUDA)
    find_package(CUDA ${CMAKE_CUDA_VERSION} REQUIRED)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    add_definitions("-D_HAS_CUDA_")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(SOLO_ENABLE_COVERAGE AND NOT WIN32)
    # Setting compiler flags for code coverage
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()

###
# SCAMP Targets
#
#   - all libraries prefixed with gauss_scamp_
#
#   - common
#   - qt_kernels
#   - gpu_kernels <- common
#   - qt_helper <- common qt_kernels cufft
#   - cpu_stats <- common
#   - cpu_kernels < - common
#   - tile <- gpu_kernels cpu_kernels qt_helper
#   - scamp <- cpu_stats tile
###

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 72 70 61 60) # 50 37 35)
endif()

add_library(gauss_scamp_common external/scamp/src/common.cpp)
if (CMAKE_CUDA_COMPILER) 
    target_link_libraries(gauss_scamp_common 
        ${CUDA_cudart_LIBRARY}
    )
endif()
set_property(TARGET gauss_scamp_common 
PROPERTY
    INTERPROCEDURAL_OPTIMIZATION TRUE 
    CXX_VISIBILITY_PRESET "hidden"
    POSITION_INDEPENDENT_CODE ON
)

add_library(gauss_scamp_qt_helper external/scamp/src/qt_helper.cpp)
if (CMAKE_CUDA_COMPILER)
    add_library(gauss_scamp_qt_kernels 
        external/scamp/src/qt_kernels.cu
    )
    set_property(TARGET gauss_scamp_qt_kernels 
        PROPERTY
            INTERPROCEDURAL_OPTIMIZATION TRUE 
            CXX_VISIBILITY_PRESET "hidden"
            POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(gauss_scamp_qt_helper 
        ${CUDA_CUFFT_LIBRARIES} 
        gauss_scamp_qt_kernels 
        gauss_scamp_common)
else()
    target_link_libraries(gauss_scamp_qt_helper gauss_scamp_common)
endif()

set_property(TARGET gauss_scamp_qt_helper 
PROPERTY
    INTERPROCEDURAL_OPTIMIZATION TRUE 
    CXX_VISIBILITY_PRESET "hidden"
    POSITION_INDEPENDENT_CODE ON
)

add_library(gauss_scamp_kernel_common external/scamp/src/kernel_common.cpp)
if (CMAKE_CUDA_COMPILER)
    add_library(gauss_scamp_gpu_utils external/scamp/src/kernel_gpu_utils.cu)
    set_property(TARGET gauss_scamp_gpu_utils 
        PROPERTY
            INTERPROCEDURAL_OPTIMIZATION TRUE 
            CXX_VISIBILITY_PRESET "hidden"
            POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(gauss_scamp_gpu_utils 
        gauss_scamp_kernel_common 
        ${CUDA_cudart_LIBRARY}
    )
    
    add_library(gauss_scamp_gpu_kernels external/scamp/src/kernels.cu)
    set_property(TARGET gauss_scamp_gpu_kernels 
        PROPERTY
            INTERPROCEDURAL_OPTIMIZATION TRUE 
            CXX_VISIBILITY_PRESET "hidden"
            POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(gauss_scamp_gpu_kernels 
        gauss_scamp_gpu_utils 
        gauss_scamp_kernel_common 
        gauss_scamp_common
    )
endif()

add_library(gauss_scamp_cpu_stats external/scamp/src/cpu_stats.cpp)
target_link_libraries(gauss_scamp_cpu_stats gauss_scamp_common)

add_library(gauss_scamp_cpu_kernels external/scamp/src/cpu_kernels.cpp)
target_link_libraries(gauss_scamp_cpu_kernels 
        gauss_scamp_kernel_common 
        gauss_scamp_common)

add_library(gauss_scamp_tile external/scamp/src/tile.cpp)
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(gauss_scamp_tile 
    gauss_scamp_common 
    gauss_scamp_gpu_kernels 
    gauss_scamp_cpu_kernels 
    gauss_scamp_qt_helper)
else()
  target_link_libraries(gauss_scamp_tile 
    gauss_scamp_common 
    gauss_scamp_cpu_kernels 
    gauss_scamp_qt_helper)
endif()

add_library(gauss_scamp_utils external/scamp/src/scamp_utils.cpp)
add_library(gauss_scamp external/scamp/src/SCAMP.cpp)
target_link_libraries(gauss_scamp
    gauss_scamp_tile 
    gauss_scamp_cpu_stats 
    gauss_scamp_common 
    gauss_scamp_qt_helper)

####
# Main algos
####

# Set the base directory to parent so src and include become equally visible
set(GAUSSLIB_BASE_DIR ${PROJECT_SOURCE_DIR})
# Define source directory
set(GAUSSLIB_SRC "${GAUSSLIB_BASE_DIR}/src")
# Define include directory
set(GAUSSLIB_INC "${GAUSSLIB_BASE_DIR}/include")

add_compile_definitions(BUILDING_GAUSS)

# Sources to add to compilation
set(GAUSSLIB_SOURCES ${GAUSSLIB_SRC}/clustering.cpp
                     ${GAUSSLIB_SRC}/dimensionality.cpp
                     ${GAUSSLIB_SRC}/distances.cpp
                     ${GAUSSLIB_SRC}/features.cpp
                     ${GAUSSLIB_SRC}/fft.cpp                     
                     ${GAUSSLIB_SRC}/filters.cpp
                     ${GAUSSLIB_SRC}/libraryInternal.cpp
                     ${GAUSSLIB_SRC}/linalg.cpp
                     ${GAUSSLIB_SRC}/matrix.cpp
                     ${GAUSSLIB_SRC}/matrixInternal.cpp
                     ${GAUSSLIB_SRC}/normalization.cpp
                     ${GAUSSLIB_SRC}/polynomial.cpp
                     ${GAUSSLIB_SRC}/random.cpp
                     ${GAUSSLIB_SRC}/regression.cpp
                     ${GAUSSLIB_SRC}/regularization.cpp
                     ${GAUSSLIB_SRC}/statistics.cpp)

# Headers to add to compilation
set(GAUSSLIB_HEADERS ${GAUSSLIB_INC}/gauss/clustering.h
                     ${GAUSSLIB_INC}/gauss/defines.h
                     ${GAUSSLIB_INC}/gauss/dimensionality.h
                     ${GAUSSLIB_INC}/gauss/distances.h
                     ${GAUSSLIB_INC}/gauss/features.h
                     ${GAUSSLIB_INC}/gauss/fft.h
                     ${GAUSSLIB_INC}/gauss/filters.h
                     ${GAUSSLIB_INC}/gauss/linalg.h
                     ${GAUSSLIB_INC}/gauss/matrix.h
                     ${GAUSSLIB_INC}/gauss/normalization.h
                     ${GAUSSLIB_INC}/gauss/polynomial.h
                     ${GAUSSLIB_INC}/gauss/regression.h
                     ${GAUSSLIB_INC}/gauss/regularization.h
                     ${GAUSSLIB_INC}/gauss/statistics.h
                     ${GAUSSLIB_INC}/gauss/internal/libraryInternal.h
                     ${GAUSSLIB_INC}/gauss/internal/matrixInternal.h
                     ${GAUSSLIB_INC}/gauss/internal/scopedHostPtr.h
                     ${GAUSSLIB_INC}/gauss/internal/vectorUtil.h)

# The output is a static library
add_library(gauss ${GAUSSLIB_HEADERS} ${GAUSSLIB_SOURCES})

target_include_directories(gauss
    PRIVATE
        ${PROJECT_SOURCE_DIR}/external
    PUBLIC
        ${ArrayFire_INCLUDE_DIRS}
        ${GAUSSLIB_INC}
        ${GAUSSLIB_INC}/internal)

# Declare compilation libraries
target_link_libraries(gauss
	PRIVATE gauss_scamp
            Eigen3::Eigen
		    Boost::boost
            ${CMAKE_DL_LIBS}
	PUBLIC  ArrayFire::af)

# Add compile definitions
target_compile_definitions(gauss
    PUBLIC GAUSS_EXPORTS
           AF_UNIFIED)    

set_target_properties(gauss
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        INTERFACE_INCLUDE_DIRECTORIES "${GAUSSLIB_INC}"
        PUBLIC_HEADER "${GAUSSLIB_INC}/gauss.h")
