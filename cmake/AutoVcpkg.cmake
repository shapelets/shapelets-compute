# ~~~
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ~~~

set(AUTO_VCPKG_GIT_REPOSITORY "https://github.com/Microsoft/vcpkg.git")
set(DOWNLOADS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.downloads)
set(DEFAULT_AUTO_VCPKG_ROOT "${CMAKE_SOURCE_DIR}/external/vcpkg")

function (vcpkg_download)
    set(vcpkg_download_contents [===[
        cmake_minimum_required(VERSION 3.5)
        project(vcpkg-download)

        include(ExternalProject)
        ExternalProject_Add(vcpkg
                    GIT_REPOSITORY @AUTO_VCPKG_GIT_REPOSITORY@
                    GIT_SHALLOW ON
                    SOURCE_DIR @AUTO_VCPKG_ROOT@
                    PATCH_COMMAND ""
                    CONFIGURE_COMMAND  ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    LOG_DOWNLOAD ON
                    LOG_CONFIGURE ON
                    LOG_INSTALL ON)
    ]===])

    string(REPLACE "@AUTO_VCPKG_GIT_REPOSITORY@" "${AUTO_VCPKG_GIT_REPOSITORY}" vcpkg_download_contents "${vcpkg_download_contents}")
    string(REPLACE "@AUTO_VCPKG_ROOT@" "${AUTO_VCPKG_ROOT}" vcpkg_download_contents "${vcpkg_download_contents}")

    file(WRITE "${DOWNLOADS_DIRECTORY}/vcpkg-download/CMakeLists.txt" "${vcpkg_download_contents}")


    execute_process(COMMAND "${CMAKE_COMMAND}"
        "-H${DOWNLOADS_DIRECTORY}/vcpkg-download"
        "-B${DOWNLOADS_DIRECTORY}/vcpkg-download")

    execute_process(COMMAND "${CMAKE_COMMAND}"
        "--build" "${DOWNLOADS_DIRECTORY}/vcpkg-download")

    if (WIN32)
        execute_process(
            COMMAND "${AUTO_VCPKG_ROOT}/bootstrap-vcpkg.bat" 
            WORKING_DIRECTORY "${AUTO_VCPKG_ROOT}"
        )
    else()
        execute_process(
            COMMAND  "${AUTO_VCPKG_ROOT}/bootstrap-vcpkg.sh" 
            WORKING_DIRECTORY "${AUTO_VCPKG_ROOT}"
        )
    endif()            

endfunction ()


function (vcpkg_install)
    # Variables:
    #   ENV{VCPKG_ROOT}         Points to an existing installation of VCPKG through environment variables.
    #   AUTO_VCPKG_ROOT         Points to an existing installation of VCPKG through command line settings in cmake
    #   AUTO_VCPKG_EXECUTABLE   Flag that indicates if vcpkg executable has been found
    #
    # We want to leave this script with 
    #   CMAKE_TOOLCHAIN_FILE    Pointing to vcpkg.cmake file for auto configuration

    # The variable controlling the process is AUTO_VCPKG_ROOT, which should be bootstrapped from the environment 
    # to find existing installations or it will be set by this process.

    # if it is not set and there is an enviornment variable point to it
    if(NOT DEFINED AUTO_VCPKG_ROOT)
        if (DEFINED ENV{VCPKG_ROOT}) 
            set(AUTO_VCPKG_ROOT "$ENV{VCPKG_ROOT}" CACHE STRING "")    
        else()
            set(AUTO_VCPKG_ROOT ${DEFAULT_AUTO_VCPKG_ROOT} CACHE STRING "")    
        endif()
    endif()
    
    # if we have the variable set, try to find the executable
    find_program(AUTO_VCPKG_EXECUTABLE vcpkg PATHS ${AUTO_VCPKG_ROOT})

    # if not found, download and/or bootstrap
    if (NOT AUTO_VCPKG_EXECUTABLE)
        message(STATUS "Couldn't find vcpkg in ${AUTO_VCPKG_ROOT}; downloading and building...")
        vcpkg_download()
    endif()

    message(STATUS "Checking for vcpkg in ${AUTO_VCPKG_ROOT}")
    find_program(AUTO_VCPKG_EXECUTABLE vcpkg PATHS ${AUTO_VCPKG_ROOT})
    if (NOT AUTO_VCPKG_EXECUTABLE)
        message(FATAL_ERROR "Cannot find vcpkg executable")
    endif ()

    mark_as_advanced(AUTO_VCPKG_EXECUTABLE)
    set(CMAKE_TOOLCHAIN_FILE "${AUTO_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

    cmake_parse_arguments(_vcpkg_install "" "TRIPLET" "" ${ARGN})
    if (NOT ARGN)
        message(STATUS "vcpkg_install() called with no packages to install")
        return()
    endif ()

    if (NOT _vcpkg_install_TRIPLET)
        set(packages ${ARGN})
    else ()
        string(APPEND ":${_vcpkg_install_TRIPLET}" packages ${ARGN})
    endif ()
    
    string(JOIN ", " join ${packages})
    message(STATUS "vcpkg_install() called to install: ${join}")

    execute_process (COMMAND "${AUTO_VCPKG_EXECUTABLE}" "install" ${packages})
endfunction ()
