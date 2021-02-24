#ifndef ALGOS_CORE_LIBRARY_H
#define ALGOS_CORE_LIBRARY_H

#include <arrayfire.h>
#include <algos/defines.h>

namespace algos {

namespace library {

typedef enum {
    KHIVA_BACKEND_DEFAULT = af::Backend::AF_BACKEND_DEFAULT,  ///< Default backend order: OpenCL -> CUDA -> CPU
    KHIVA_BACKEND_CPU = af::Backend::AF_BACKEND_CPU,          ///< CPU a.k.a sequential algorithms
    KHIVA_BACKEND_CUDA = af::Backend::AF_BACKEND_CUDA,        ///< CUDA Compute Backend
    KHIVA_BACKEND_OPENCL = af::Backend::AF_BACKEND_OPENCL,    ///< OpenCL Compute Backend
} khiva_backend;

typedef khiva_backend Backend;

/**
 * @brief Get information from the active backend.
 *
 * @return std::string The information of the backend.
 */
ALGOSAPI std::string backendInfo();

/**
 * @brief Set the backend.
 *
 * @param be The desired backend.
 */
ALGOSAPI void setBackend(algos::library::Backend be);

/**
 * @brief Get the active backend.
 *
 * @return algos::library::Backend The active backend.
 */
ALGOSAPI algos::library::Backend getBackend();

/**
 * @brief Get the available backends.
 *
 * @return int The available backends.
 */
ALGOSAPI int getBackends();

/**
 * @brief Set the device.
 *
 * @param device The desired device.
 */
ALGOSAPI void setDevice(int device);

/**
 * @brief Get the active device.
 *
 * @return int The active device.
 */
ALGOSAPI int getDevice();

/**
 * @brief Get the device count.
 *
 * @return int The device count.
 */
ALGOSAPI int getDeviceCount();

/**
 * @brief Set the memory of the device in use. This information is used for splitting some algorithms and execute them
 * in batch mode. The default value used if it is not set is 4GB.
 *
 * @param memory The device memory.
 */
ALGOSAPI void setDeviceMemoryInGB(double memory);

}  // namespace library
}  // namespace algos

#endif
