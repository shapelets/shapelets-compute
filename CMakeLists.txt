cmake_minimum_required(VERSION 3.9.6)

# Load from environment variables
list(APPEND CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# VCPKG to install dependencies
include(AutoVcpkg)
vcpkg_install(pybind11 gtest eigen3 benchmark boost spdlog)

# ArrayFire is a different download
if (APPLE)
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/etc/macos_arrayfire.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/etc"
        RESULT_VARIABLE AF_DOWNLOADED
    )
elseif (UNIX)
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/etc/linux_arrayfire.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/etc"
        RESULT_VARIABLE AF_DOWNLOADED
    )
elseif (WIN32)
    message(FATAL_ERROR "TODO: Missing Win32 download script")
else()
    message(FATAL_ERROR "Unable to download arrayfire as platform is not recognised")
endif()

