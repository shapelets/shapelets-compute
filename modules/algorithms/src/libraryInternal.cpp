#include "algos/internal/libraryInternal.h"

#include <cmath>

namespace {
double currentDeviceMemoryInGB = 4.0;
double defaultMemoryInGB = 4.0;
}  // namespace

namespace algos {
namespace library {
namespace internal {

void setDeviceMemoryInGB(double memory) { currentDeviceMemoryInGB = memory; }

long getValueScaledToMemoryDevice(long value, Complexity complexity) {
    double ratio = currentDeviceMemoryInGB / defaultMemoryInGB;
    long newValue = value;
    switch (complexity) {
        case Complexity::LINEAR:
            newValue = static_cast<long>(static_cast<double>(value) * ratio);
            break;
        case Complexity::CUADRATIC:
            newValue = static_cast<long>(static_cast<double>(value) * std::sqrt(ratio));
            break;
        case Complexity::CUBIC:
            newValue = static_cast<long>(static_cast<double>(value) * std::cbrt(ratio));
            break;
    }
    return newValue;
}

}  // namespace internal
}  // namespace library
}  // namespace algos
