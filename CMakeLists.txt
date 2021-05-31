# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.9.6)

####
# START: Pre-Project Setup
####
set(VCPKG_FEATURE_FLAGS versions,manifests CACHE STRING "")

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
else()
    set(VCPKG_TARGET_ARCHITECTURE "x64" CACHE STRING "" )
    SET(VCPKG_LIBRARY_LINKAGE "static" CACHE STRING "")
endif()

# Load from environment variables
list(APPEND CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    include(AutoVcpkg)
    vcpkg_install()
endif()

# Installs a local copy of array fire if not present in the system.
include(LocalArrayFire)

####
# END: Pre-Project Setup
####

project(ShapeletsSolo)


set(CMAKE_SKIP_BUILD_RPATH  FALSE)
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_BUILD_RPATH "@loader_path/.libs")
    set(CMAKE_INSTALL_RPATH "@loader_path/.libs")
else()
    set(CMAKE_BUILD_RPATH "$ORIGIN/.libs")
    set(CMAKE_INSTALL_RPATH "$ORIGIN/.libs")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)



# Options
option(SOLO_ENABLE_COVERAGE "This flag anables Code Coverage of the code" OFF)

# Using C++ 11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Required
set(CMAKE_CXX_EXTENSIONS ON) # with compiler extensions like gnu++11

# Set the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE)
endif()

# In debug mode, avoid any optimisations
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "-g -O0")
endif()

# Avoid conflicts with min and max functions
if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

#global settings
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Turn compiler warnings up to 11
    string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -pedantic -Wno-gnu-zero-variadic-macro-arguments -Wint-conversion -Wshorten-64-to-32")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -pedantic")
endif()

add_subdirectory(external/pybind11)
add_subdirectory(modules/gauss)
add_subdirectory(modules/pygauss)