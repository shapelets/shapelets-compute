
function (download_af)
    
    execute_process(
        COMMAND python3 setup_af.py
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake"
        RESULT_VARIABLE AF_DOWNLOADED
    )

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
else()
    message(STATUS "ArrayFire: No system-wide installation found")
endif()

if (NOT ArrayFire_FOUND)
    # check if a download is required
    if (NOT (EXISTS "${CMAKE_SOURCE_DIR}/external/arrayfire/include/arrayfire.h"))
        download_af()
    else()
        message(STATUS "ArrayFire: Found project installation at ./external/arrayfire")
    endif()

    # ensure ArrayFire_DIR is set
    
    if (WIN32)
        set(ArrayFire_DIR "${CMAKE_SOURCE_DIR}/external/arrayfire/cmake")
    else()
        set(ArrayFire_DIR "${CMAKE_SOURCE_DIR}/external/arrayfire/share/ArrayFire/cmake")
    endif()    
    
    # add directory to list of cmake modules
    list(APPEND CMAKE_PREFIX_PATH ${ArrayFire_DIR})
endif()

# report where AF is located
message(STATUS "ArrayFire: Using installation at ${ArrayFire_DIR}")
