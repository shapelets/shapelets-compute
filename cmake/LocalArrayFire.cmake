
OPTION(ArrayFire_Local "Chooses local ArrayFire copy over installed copy" ON)

function (download_af)
    
    if (APPLE)
        # Download and unpack apple libraries to lib/arrayfire
        message(STATUS "ArrayFire: Starting download for Mac")
        execute_process(
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macos_arrayfire.sh ${CMAKE_SOURCE_DIR}/external/arrayfire
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
            COMMAND_ECHO STDERR
            RESULT_VARIABLE AF_DOWNLOADED
        )

    elseif (UNIX)
        # Download and unpack linux libraries to lib/arrayfire
        message(STATUS "ArrayFire: Starting download for Unix")
        
        execute_process(
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux_arrayfire.sh ${CMAKE_SOURCE_DIR}/external/arrayfire
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
            COMMAND_ECHO STDERR
            RESULT_VARIABLE AF_DOWNLOADED
        )

    elseif (WIN32)
        message(STATUS "ArrayFire: Starting download for Windows")
        # Download and unpack windows libraries to lib/arrayfire
        message(FATAL_ERROR "TODO: Missing Win32 download script")

    else()
        message(FATAL_ERROR "ArrayFire: Unknown platform.")

    endif()

    if(AF_DOWNLOADED AND NOT AF_DOWNLOADED EQUAL 0)
        message(FATAL_ERROR "ArrayFire: Unsucessful download")
    else()
        message(STATUS "ArrayFire: local copy downloaded")
    endif()
endfunction()

#determine if a system copy exists
find_package(ArrayFire QUIET)

if (ArrayFire_FOUND)
    message(STATUS "ArrayFire: System Installation at ${ArrayFire_DIR}")
endif()

# if a local copy is preferred or there is no installation of 
# arrayfire passed through command line arguments or, lastly, 
# there is no system installation of array fire...
if (ArrayFire_Local OR NOT EXISTS ${ArrayFire_DIR} OR NOT ArrayFire_FOUND)
    # check if a download is required
    if (NOT (EXISTS "${CMAKE_SOURCE_DIR}/external/arrayfire/include/arrayfire.h"))
        download_af()
    else()
        message(STATUS "ArrayFire: Found project installation at ./external/arrayfire")
    endif()

    # ensure ArrayFire_DIR is set
    set(ArrayFire_DIR "${CMAKE_SOURCE_DIR}/external/arrayfire/share/ArrayFire/cmake")
    # add directory to list of cmake modules
    list(APPEND CMAKE_MODULE_PATH ${ArrayFire_DIR})
endif()

# report where AF is located
message(STATUS "ArrayFire: Using installation at ./external/arrayfire")
