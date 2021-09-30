# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.4...3.18)

project(PyGauss)

# Find the ArrayFire package.
find_package(ArrayFire CONFIG REQUIRED)

# Find arrow
# find_package(Arrow CONFIG REQUIRED)

set(PYGAUSS_BASE_DIR "${PROJECT_SOURCE_DIR}")
set(PYGAUSS_SRC "${PYGAUSS_BASE_DIR}/src")
set(PYGAUSS_INC "${PYGAUSS_BASE_DIR}/include")
set(PYGAUSS_LOG_INC "${PYGAUSS_BASE_DIR}/../../external/spdlog/include")

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
        ${PYGAUSS_LOG_INC}
        ${PYGAUSS_INC})

# Add compile definitions
target_compile_definitions(PyGauss PRIVATE AF_UNIFIED)

target_link_libraries(PyGauss
        PUBLIC 
                ArrayFire::af
        PRIVATE
                gauss
                $<$<BOOL:${CMAKE_CUDA_COMPILER}>:cufft>
                $<$<BOOL:${CMAKE_CUDA_COMPILER}>:cudart>
        # spdlog::spdlog_header_only
                ${CMAKE_DL_LIBS})

add_custom_command(TARGET PyGauss POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:PyGauss>/.libs)  

get_property(AFLIBIMPCFG TARGET ArrayFire::af PROPERTY IMPORTED_CONFIGURATIONS)
get_property(AFLIB TARGET ArrayFire::af PROPERTY IMPORTED_LOCATION_${AFLIBIMPCFG})
get_filename_component(AFLIBDIR "${AFLIB}" DIRECTORY)

IF (APPLE)
        if (COPY_ALL_FILES)
                set(RUNTIME_LIBS 
                        ${AFLIBDIR}/libaf.3.dylib 
                        ${AFLIBDIR}/libafcpu.3.dylib
                        ${AFLIBDIR}/libafcuda.3.dylib
                        ${AFLIBDIR}/libafopencl.3.dylib
                        ${AFLIBDIR}/libcudart.10.1.dylib
                        ${AFLIBDIR}/libcudnn.dylib
                        ${AFLIBDIR}/libiomp5.dylib
                        ${AFLIBDIR}/libiompstubs5.dylib
                        ${AFLIBDIR}/libmkl_avx.dylib
                        ${AFLIBDIR}/libmkl_avx2.dylib
                        ${AFLIBDIR}/libmkl_avx512.dylib
                        ${AFLIBDIR}/libmkl_core.dylib
                        ${AFLIBDIR}/libmkl_intel_lp64.dylib
                        ${AFLIBDIR}/libmkl_intel_thread.dylib
                        ${AFLIBDIR}/libmkl_mc.dylib
                        ${AFLIBDIR}/libmkl_mc3.dylib
                        ${AFLIBDIR}/libnvrtc-builtins.dylib
                        ${AFLIBDIR}/libnvrtc.10.1.dylib)
        else()
                set(RUNTIME_LIBS 
                        ${AFLIBDIR}/libaf.3.dylib
			${AFLIBDIR}/libiomp5.dylib
			${AFLIBDIR}/libmkl_intel_thread.dylib
			${AFLIBDIR}/libmkl_intel_lp64.dylib
			${AFLIBDIR}/libmkl_core.dylib)
        endif()

elseif(UNIX)
        if (COPY_ALL_FILES)
                set(RUNTIME_LIBS 
                        ${AFLIBDIR}/libaf.so.3
                        ${AFLIBDIR}/libafcpu.so.3
                        ${AFLIBDIR}/libafcuda.so.3
                        ${AFLIBDIR}/libafopencl.so.3
                        ${AFLIBDIR}/libcudnn.so.7.6.5
                        ${AFLIBDIR}/libiomp5.so
                        ${AFLIBDIR}/libmkl_avx.so
                        ${AFLIBDIR}/libmkl_avx2.so
                        ${AFLIBDIR}/libmkl_avx512.so
                        ${AFLIBDIR}/libmkl_core.so
                        ${AFLIBDIR}/libmkl_def.so
                        ${AFLIBDIR}/libmkl_intel_lp64.so
                        ${AFLIBDIR}/libmkl_intel_thread.so
                        ${AFLIBDIR}/libmkl_mc.so
                        ${AFLIBDIR}/libmkl_mc3.so
                        ${AFLIBDIR}/libnvrtc-builtins.so
                        ${AFLIBDIR}/libnvrtc.so.11.1
                        ${AFLIBDIR}/libOpenCL.so.1)
        else()
                set(RUNTIME_LIBS 
                        ${AFLIBDIR}/libaf.so.3
                        ${AFLIBDIR}/libmkl_core.so
                        ${AFLIBDIR}/libiomp5.so
                        ${AFLIBDIR}/libmkl_intel_lp64.so
                        ${AFLIBDIR}/libmkl_intel_thread.so)
        endif()                        
else()
        if (COPY_ALL_FILES)
                file(GLOB RUNTIME_LIBS ${AFLIBDIR}/*.dll)
        else()
                set(RUNTIME_LIBS ${AFLIBDIR}/af.dll)                
        endif()                        
endif()

foreach(file_i ${RUNTIME_LIBS})
        add_custom_command(TARGET PyGauss POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${file_i}
        $<TARGET_FILE_DIR:PyGauss>/.libs)
endforeach(file_i)

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




# # Get all propreties that cmake supports
# execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

# # Convert command output into a CMake list
# STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
# STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

# function(print_properties)
#     message ("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
# endfunction(print_properties)

# function(print_target_properties tgt)
#     if(NOT TARGET ${tgt})
#       message("There is no target named '${tgt}'")
#       return()
#     endif()

#     foreach (prop ${CMAKE_PROPERTY_LIST})
#         string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
#     # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
#     if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
#         continue()
#     endif()
#         # message ("Checking ${prop}")
#         get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
#         if (propval)
#             get_target_property(propval ${tgt} ${prop})
#             message ("${tgt} ${prop} = ${propval}")
#         endif()
#     endforeach(prop)
# endfunction(print_target_properties)

# print_target_properties(ArrayFire::af)
