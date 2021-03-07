cmake_minimum_required(VERSION 3.9.6)

project(Gauss)

#sanity check
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/scamp/CMakeLists.txt")
	message(FATAL_ERROR "git submodule was not initialized.")
endif()

find_package(ArrayFire CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

#find_package(CUDA )
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
	message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
	enable_language(CUDA)
    add_definitions("-D_HAS_CUDA_")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(CMAKE_CUDA_STANDARD 14)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_70")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")    
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
add_library(gauss_scamp_common external/scamp/src/common.cpp)
set_target_properties(gauss_scamp_common
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CUDA_COMPILER)
    add_library(gauss_scamp_qt_kernels external/scamp/src/qt_kernels.cu)
    set_target_properties(gauss_scamp_qt_kernels
        PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
            CXX_VISIBILITY_PRESET "hidden"
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}") 

    add_library(gauss_scamp_gpu_kernels external/scamp/src/kernels.cu)
    target_link_libraries(gauss_scamp_gpu_kernels gauss_scamp_common)
    set_target_properties(gauss_scamp_gpu_kernels
        PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
            CXX_VISIBILITY_PRESET "hidden"
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}") 
endif()

add_library(gauss_scamp_qt_helper external/scamp/src/qt_helper.cpp)
set_target_properties(gauss_scamp_qt_helper
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gauss_scamp_qt_helper
    gauss_scamp_common
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:cufft>
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:gauss_scamp_qt_kernels>)

add_library(gauss_scamp_cpu_stats external/scamp/src/cpu_stats.cpp)
set_target_properties(gauss_scamp_cpu_stats
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gauss_scamp_cpu_stats gauss_scamp_common)

add_library(gauss_scamp_cpu_kernels external/scamp/src/cpu_kernels.cpp)
set_target_properties(gauss_scamp_cpu_kernels
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gauss_scamp_cpu_kernels gauss_scamp_common)

add_library(gauss_scamp_tile external/scamp/src/tile.cpp)
set_target_properties(gauss_scamp_tile
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gauss_scamp_tile
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:gauss_scamp_gpu_kernels>
    gauss_scamp_cpu_kernels
    gauss_scamp_qt_helper)

add_library(gauss_scamp external/scamp/src/SCAMP.cpp)
set_target_properties(gauss_scamp
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gauss_scamp
    PRIVATE
        gauss_scamp_tile
        gauss_scamp_cpu_stats)

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
                     ${GAUSSLIB_SRC}/statistics.cpp
                     ${GAUSSLIB_SRC}/util.cpp)

# Headers to add to compilation
set(GAUSSLIB_HEADERS ${GAUSSLIB_INC}/gauss/clustering.h
                     ${GAUSSLIB_INC}/gauss/defines.h
                     ${GAUSSLIB_INC}/gauss/dimensionality.h
                     ${GAUSSLIB_INC}/gauss/distances.h
                     ${GAUSSLIB_INC}/gauss/features.h
                     ${GAUSSLIB_INC}/gauss/filters.h
                     ${GAUSSLIB_INC}/gauss/linalg.h
                     ${GAUSSLIB_INC}/gauss/matrix.h
                     ${GAUSSLIB_INC}/gauss/normalization.h
                     ${GAUSSLIB_INC}/gauss/polynomial.h
                     ${GAUSSLIB_INC}/gauss/regression.h
                     ${GAUSSLIB_INC}/gauss/regularization.h
                     ${GAUSSLIB_INC}/gauss/statistics.h
                     ${GAUSSLIB_INC}/gauss/utils.h
                     ${GAUSSLIB_INC}/gauss/internal/libraryInternal.h
                     ${GAUSSLIB_INC}/gauss/internal/matrixInternal.h
                     ${GAUSSLIB_INC}/gauss/internal/scopedHostPtr.h
                     ${GAUSSLIB_INC}/gauss/internal/util.h
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
    PRIVATE GAUSS_EXPORTS
    PUBLIC AF_UNIFIED)    

set_target_properties(gauss
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
        CXX_VISIBILITY_PRESET "hidden"
        INTERFACE_INCLUDE_DIRECTORIES "${GAUSSLIB_INC}"
        PUBLIC_HEADER "${GAUSSLIB_INC}/gauss.h")
