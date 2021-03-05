#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <af_array/af_array.h>

namespace spd = spdlog;
namespace py = pybind11;

/**
 * Binds array fire to python.  Since this is a large binding, it is split in several files
 * all of them named with the pattern af_array_XXXXX_bindings.
 *
 * @param m  The module where the array definitions are to take place.
 */
void af_array_bindings(py::module_ &m) {

    m.def("manual_eval_enabled",
          []() {
        return af::getManualEvalFlag();
    },
    "Informs if computations would only be triggered when a eval is directly requested.");

    m.def("enable_manual_eval",
          [](const bool& newValue) {
              spd::debug("Manual Evaluation is now {}", newValue ? "ENABLED" : "DISABLED");
              return af::setManualEvalFlag(newValue);
          },
          "Changes the way results are computed.  "
          "\n"
          "When manually evaluation is disabled, the system will compute as soon as possible, "
          "reducing the opportunities for kernel fusion; however, when manual evaluation is "
          "enabled, computations have a better chance of merging, resulting in a far more "
          "effective computation.  When enabled, results should be requested through `eval` "
          "methods and `sync` to ensure device's work queue is completed.");

    device_bindings(m);
    enum_bindings(m);
    array_obj_bindings(m);
    construction_bindings(m);
    random_bindings(m);
    informative_bindings(m);
    extract_transform_bindings(m);
    linear_algebra_bindings(m);
    complex_bindings(m);
    logic_bindings(m);
    math_bindings(m);
    statistic_bindings(m);
    algorithm_bindings(m);
    batch_bindings(m);
    signal_processing_bindings(m);

}
