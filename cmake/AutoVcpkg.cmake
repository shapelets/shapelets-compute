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

    # if it is not set and there is an enviornment variable point to it
    if(NOT DEFINED AUTO_VCPKG_ROOT)
        if (DEFINED ENV{VCPKG_ROOT}) 
            set(AUTO_VCPKG_ROOT "$ENV{VCPKG_ROOT}" CACHE STRING "")    
        elseif(DEFINED ENV{VCPKG_INSTALLATION_ROOT})
            set(AUTO_VCPKG_ROOT "$ENV{VCPKG_INSTALLATION_ROOT}" CACHE STRING "")    
        else()
            set(AUTO_VCPKG_ROOT ${DEFAULT_AUTO_VCPKG_ROOT} CACHE STRING "")    
        endif()
    endif()
    
    # if we have the variable set, try to find the executable
    find_program(AUTO_VCPKG_EXECUTABLE vcpkg PATHS ${AUTO_VCPKG_ROOT} 
        NO_DEFAULT_PATH 
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH)

    # if not found, download and/or bootstrap
    if (NOT AUTO_VCPKG_EXECUTABLE)
        message(STATUS "Couldn't find vcpkg in ${AUTO_VCPKG_ROOT}; downloading and building...")
        vcpkg_download()
        find_program(AUTO_VCPKG_EXECUTABLE vcpkg PATHS ${AUTO_VCPKG_ROOT}
            NO_DEFAULT_PATH 
            NO_CMAKE_ENVIRONMENT_PATH 
            NO_SYSTEM_ENVIRONMENT_PATH)

        if (NOT AUTO_VCPKG_EXECUTABLE)
            message(FATAL_ERROR "Cannot find vcpkg executable")
        endif ()
    endif()

    mark_as_advanced(AUTO_VCPKG_EXECUTABLE)
    set(CMAKE_TOOLCHAIN_FILE "${AUTO_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

    execute_process (COMMAND "${AUTO_VCPKG_EXECUTABLE} install --feature-flags:versions,manifest")
endfunction ()
