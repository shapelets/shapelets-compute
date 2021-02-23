cmake_minimum_required(VERSION 3.9.6)

project(ShapeletsSolo)

# Load from environment variables
list(APPEND CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Installs a local copy of array fire if not present in the system.
include(LocalArrayFire) 

# VCPKG to install dependencies
include(AutoVcpkg)
vcpkg_install(pybind11 gtest eigen3 benchmark boost spdlog)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

