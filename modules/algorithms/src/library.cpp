#include <arrayfire.h>
#include <algos/library.h>

#include "algos/internal/libraryInternal.h"

std::string algos::library::backendInfo() { return af::infoString(); }

void algos::library::setBackend(algos::library::Backend be) { af::setBackend(static_cast<af::Backend>(be)); }

algos::library::Backend algos::library::getBackend() {
    return static_cast<algos::library::Backend>(af::getActiveBackend());
}

int algos::library::getBackends() { return af::getAvailableBackends(); }

void algos::library::setDevice(int device) { af::setDevice(device); }

int algos::library::getDevice() { return af::getDevice(); }

int algos::library::getDeviceCount() { return af::getDeviceCount(); }

void algos::library::setDeviceMemoryInGB(double memory) { algos::library::internal::setDeviceMemoryInGB(memory); }
