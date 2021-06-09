/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include "pygauss.h"

namespace py = pybind11;
namespace spd = spdlog;

using namespace pygauss::bindings;


PYBIND11_MODULE(_pygauss, m) {

    // spd::set_level(spd::level::level_enum::debug);

#ifndef NDEBUG
    spd::set_level(spd::level::level_enum::debug);
#endif

    m.def("manual_eval_enabled",
          []() {
              return af::getManualEvalFlag();
          });

    m.def("enable_manual_eval",
          [](const bool &new_value) {
              spd::debug("Manual Evaluation is now {}", new_value ? "ENABLED" : "DISABLED");
              return af::setManualEvalFlag(new_value);
          },
          py::arg("new_value").none(false));

    m.def("af_version",
        []() {
            return AF_API_VERSION_CURRENT;
        });

    shared_enum_types(m);
    device_operations(m);

    array_obj(m);

    batch_api(m);

    parallel_algorithms(m);
    array_construction_operations(m);
    extract_and_transform_operations(m);
    linear_algebra_operations(m);
    logic_operations(m);
    math_operations(m);
    random_numbers(m);
    signal_processing_functions(m);
    statistic_functions(m);
    matrix_profile_functions(m);
    gauss_distance_functions(m);
    gauss_normalization_functions(m);
    gauss_dimensionality_functions(m);
    clustering_functions(m);
}
