/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_PRIVATE_HOST_PTR_H
#define GAUSS_PRIVATE_HOST_PTR_H

#ifndef BUILDING_GAUSS
#error Internal headers cannot be included from user code
#endif

#include <arrayfire.h>

#include <memory>

namespace gauss::utils {

template <typename T>
using ScopedHostPtr = std::unique_ptr<T[], decltype(&af::freeHost)>;

template <typename T>
ScopedHostPtr<T> makeScopedHostPtr(T *ptr) {
    return ScopedHostPtr<T>(ptr, &af::freeHost);
}
}  // namespace gauss

#endif