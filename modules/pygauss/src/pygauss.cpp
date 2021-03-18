
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include "pygauss.h"

namespace py = pybind11;
namespace spd = spdlog;

using namespace pygauss::bindings;


PYBIND11_MODULE(_pygauss, m) {

    // spd::set_level(spd::level::level_enum::debug);

#ifndef NDEBUG
    // spd::set_level(spd::level::level_enum::debug);
#endif

    m.def("manual_eval_enabled",
          []() {
              return af::getManualEvalFlag();
          },
          "Informs if computations would only be triggered when a eval is directly requested.");

    m.def("enable_manual_eval",
          [](const bool &new_value) {
              spd::debug("Manual Evaluation is now {}", new_value ? "ENABLED" : "DISABLED");
              return af::setManualEvalFlag(new_value);
          },
          py::arg("new_value").none(false),
          "Changes the way results are computed.  "
          "\n"
          "When manually evaluation is disabled, the system will compute as soon as possible, "
          "reducing the opportunities for kernel fusion; however, when manual evaluation is "
          "enabled, computations have a better chance of merging, resulting in a far more "
          "effective computation.  When enabled, results should be requested through `eval` "
          "methods and `sync` to ensure device's work queue is completed.");

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
    gauss_statistic_bindings(m);
    gauss_normalization_functions(m);
}
