cmake_minimum_required(VERSION 3.9.6)

project(ALGOLIB)

#sanity check
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/scamp/CMakeLists.txt")
	message(FATAL_ERROR "git submodule was not initialized.")
endif()

# Using C++ 17
set(CMAKE_CXX_STANDARD 17)
# Required
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
# with compiler extensions like gnu++11
set(CMAKE_CXX_EXTENSIONS ON) 

find_package(ArrayFire CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

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
###
add_library(common external/scamp/src/common.cpp)
set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON) 

if (CMAKE_CUDA_COMPILER)
    add_library(qt_kernels external/scamp/src/qt_kernels.cu)
    set_target_properties(qt_kernels
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}") 

    add_library(gpu_kernels external/scamp/src/kernels.cu)
    target_link_libraries(gpu_kernels common)
    set_target_properties(gpu_kernels
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}") 
endif()

add_library(qt_helper external/scamp/src/qt_helper.cpp) 
set_target_properties(qt_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(qt_helper 
	common
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:cufft>
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:qt_kernels>)

add_library(cpu_stats external/scamp/src/cpu_stats.cpp)
set_target_properties(cpu_stats PROPERTIES POSITION_INDEPENDENT_CODE ON) 
target_link_libraries(cpu_stats common)

add_library(cpu_kernels external/scamp/src/cpu_kernels.cpp)
set_target_properties(cpu_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON) 
target_link_libraries(cpu_kernels common)

add_library(tile external/scamp/src/tile.cpp) 
set_target_properties(tile PROPERTIES POSITION_INDEPENDENT_CODE ON) 
target_link_libraries(tile 
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:gpu_kernels>
    cpu_kernels 
    qt_helper) 

add_library(scamp external/scamp/src/SCAMP.cpp) 
set_target_properties(scamp PROPERTIES POSITION_INDEPENDENT_CODE ON) 
target_link_libraries(scamp 
    PRIVATE tile 
            cpu_stats)    

####
# Main algos
####

# Set the base directory to parent so src and include become equaly visible
set(ALGOLIB_BASE_DIR ${PROJECT_SOURCE_DIR})
# Define source directory
set(ALGOLIB_SRC "${ALGOLIB_BASE_DIR}/src")
# Define include directory
set(ALGOLIB_INC "${ALGOLIB_BASE_DIR}/include")

add_compile_definitions(BUILDING_ALGOS)

# Sources to add to compilation
set(ALGOLIB_SOURCES ${ALGOLIB_SRC}/clustering.cpp
                    ${ALGOLIB_SRC}/dimensionality.cpp
                    ${ALGOLIB_SRC}/distances.cpp
                    ${ALGOLIB_SRC}/features.cpp
                    ${ALGOLIB_SRC}/libraryInternal.cpp
                    ${ALGOLIB_SRC}/linalg.cpp
                    ${ALGOLIB_SRC}/matrix.cpp
                    ${ALGOLIB_SRC}/matrixInternal.cpp
                    ${ALGOLIB_SRC}/normalization.cpp
                    ${ALGOLIB_SRC}/polynomial.cpp
                    ${ALGOLIB_SRC}/regression.cpp
                    ${ALGOLIB_SRC}/regularization.cpp
                    ${ALGOLIB_SRC}/statistics.cpp
                    ${ALGOLIB_SRC}/util.cpp)

# Headers to add to compilation
set(ALGOLIB_HEADERS ${ALGOLIB_INC}/algos/clustering.h
                    ${ALGOLIB_INC}/algos/defines.h
                    ${ALGOLIB_INC}/algos/dimensionality.h
                    ${ALGOLIB_INC}/algos/distances.h
                    ${ALGOLIB_INC}/algos/features.h
                    ${ALGOLIB_INC}/algos/linalg.h
                    ${ALGOLIB_INC}/algos/matrix.h
                    ${ALGOLIB_INC}/algos/normalization.h
                    ${ALGOLIB_INC}/algos/polynomial.h
                    ${ALGOLIB_INC}/algos/regression.h
                    ${ALGOLIB_INC}/algos/regularization.h
                    ${ALGOLIB_INC}/algos/statistics.h
                    ${ALGOLIB_INC}/algos/utils.h
                    ${ALGOLIB_INC}/algos/internal/libraryInternal.h
                    ${ALGOLIB_INC}/algos/internal/matrixInternal.h
                    ${ALGOLIB_INC}/algos/internal/scopedHostPtr.h
                    ${ALGOLIB_INC}/algos/internal/util.h
                    ${ALGOLIB_INC}/algos/internal/vectorUtil.h)

set(ALGOLIB algorithms)

# The output is a static library
add_library(${ALGOLIB} ${ALGOLIB_HEADERS} ${ALGOLIB_SOURCES})

target_include_directories(${ALGOLIB} 
    PRIVATE   
        ${ArrayFire_INCLUDE_DIRS}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/external
        ${ALGOLIB_INC}
        ${ALGOLIB_INC}/internal)

# Declare compilation libraries
target_link_libraries(${ALGOLIB}
	PRIVATE scamp
            Eigen3::Eigen
		    Boost::boost
            ${CMAKE_DL_LIBS}
	PUBLIC  ArrayFire::af)

# Add compile definitions
target_compile_definitions(${ALGOLIB} 
    PRIVATE ALGO_EXPORTS 
    PUBLIC AF_UNIFIED)    

set_target_properties(${ALGOLIB} 
    PROPERTIES 
        PUBLIC_HEADER "${ALGOLIB_INC}/algos.h")
        
# export(EXPORT Khiva
#        NAMESPACE algos::
#        FILE cmake/KhivaTargets.cmake)        

