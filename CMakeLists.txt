cmake_minimum_required(VERSION 3.9.6)

####
# START: Pre-Project Setup
####

# Load from environment variables
list(APPEND CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Installs a local copy of array fire if not present in the system.
include(LocalArrayFire)

# VCPKG to install dependencies
include(AutoVcpkg)
vcpkg_install(pybind11 gtest eigen3 benchmark boost spdlog)

####
# END: Pre-Project Setup
####


project(ShapeletsSolo)

# Options
option(SOLO_ENABLE_COVERAGE "This flag anables Code Coverage of the code" OFF)

# Using C++ 11
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON) # Required
SET(CMAKE_CXX_EXTENSIONS ON) # with compiler extensions like gnu++11

# Set the build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" 
      CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE)
ENDIF()

# In debug mode, avoid any optimisations
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    SET(CMAKE_CXX_FLAGS "-g -O0")
ENDIF()

# Avoid conflicts with min and max functions
IF(WIN32)
    ADD_DEFINITIONS(-DNOMINMAX)
ENDIF()

#global settings
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Turn compiler warnings up to 11
    string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -pedantic -Wno-gnu-zero-variadic-macro-arguments -Wint-conversion -Wshorten-64-to-32")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra -pedantic")
endif()


add_subdirectory(modules/gauss)
add_subdirectory(modules/python)