cmake_minimum_required(VERSION 3.4...3.18)

project(PyGauss)



# Find the ArrayFire package.
find_package(ArrayFire CONFIG REQUIRED)

# set_target_properties(ArrayFire PROPERTIES
#         MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo
#         MAP_IMPORTED_CONFIG_DEBUG RelWithDebInfo
#         MAP_IMPORTED_CONFIG_RELWITHDEBUGINFO RelWithDebInfo)

# Logging support
# find_package(spdlog CONFIG REQUIRED)

# Find the pybind11 package
# find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)
# find_package(pybind11 REQUIRED)

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
        add_custom_command(TARGET PyGauss POST_BUILD
                COMMAND cp ARGS -R
                ${AFLIBDIR}/
                $<TARGET_FILE_DIR:PyGauss>/.libs)
elseif(UNIX)
        add_custom_command(TARGET PyGauss POST_BUILD
                COMMAND cp ARGS -P
                ${AFLIBDIR}/*
                $<TARGET_FILE_DIR:PyGauss>/.libs)
        add_custom_command(TARGET PyGauss POST_BUILD
                COMMAND rm
                $<TARGET_FILE_DIR:PyGauss>/.libs/*.debug)                
else()
        file(GLOB AF_DEPS ${AFLIBDIR}/*.dll)
        foreach(file_i ${AF_DEPS})
                message("Copying ${file_i}...")
                add_custom_command(TARGET PyGauss POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${file_i}
                        $<TARGET_FILE_DIR:PyGauss>/.libs)
        endforeach(file_i)
endif()

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