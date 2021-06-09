/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_LIBRARY_INTERNAL_H
#define GAUSS_LIBRARY_INTERNAL_H

#ifndef BUILDING_GAUSS
#error Internal headers cannot be included from user code
#endif

#include <gauss/defines.h>

namespace gauss::library::internal {

enum class Complexity { LINEAR, CUADRATIC, CUBIC };

/**
 * @brief Set the memory of the device in use. This information is used for splitting some algorithms and execute them
 * in batch mode. The default value used if it is not set is 4GB.
 *
 * @param memory The device memory.
 */
GAUSSAPI void setDeviceMemoryInGB(double memory);

/**
 * @brief Get the value scaled to the memory of the device taking into account the Memory complexity.
 *
 * @param value The value to scale.
 * @param complexity The complexity to scale with.
 *
 * @return the scaled value.
 */
GAUSSAPI long getValueScaledToMemoryDevice(long value, Complexity complexity);

}  // namespace gauss

#endif
