// Copyright (c) 2019 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef ALGOS_CORE_LIBRARY_INTERNAL_H
#define ALGOS_CORE_LIBRARY_INTERNAL_H

#ifndef BUILDING_ALGOS
#error Internal headers cannot be included from user code
#endif

#include <algos/defines.h>

namespace algos {
namespace library {
namespace internal {

enum class Complexity { LINEAR, CUADRATIC, CUBIC };

/**
 * @brief Set the memory of the device in use. This information is used for splitting some algorithms and execute them
 * in batch mode. The default value used if it is not set is 4GB.
 *
 * @param memory The device memory.
 */
ALGOSAPI void setDeviceMemoryInGB(double memory);

/**
 * @brief Get the value scaled to the memory of the device taking into account the Memory complexity.
 *
 * @param value The value to scale.
 * @param complexity The complexity to scale with.
 *
 * @return the scaled value.
 */
ALGOSAPI long getValueScaledToMemoryDevice(long value, Complexity complexity);

}  // namespace internal
}  // namespace library
}  // namespace algos

#endif
