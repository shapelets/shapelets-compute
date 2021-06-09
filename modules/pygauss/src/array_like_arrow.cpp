/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

namespace py = pybind11;

namespace pygauss::arraylike::detail {

    bool is_arrow(const py::object &value) {
        return false;
    }

    std::optional<af::array> from_arrow(const py::object &value) {
        return std::nullopt;
    }
}
