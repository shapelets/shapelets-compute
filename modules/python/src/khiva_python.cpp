
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include "khiva_python.h"

namespace py = pybind11;
namespace spd = spdlog;

PYBIND11_MODULE(khiva, m) {
    m.doc() = R"__d__(
        Khiva module
        ------------

        .. currentmodule:: khiva

        .. autosummary::
            :toctree: _generate
        
    )__d__";

#ifndef NDEBUG
    spd::set_level(spd::level::level_enum::debug);
#endif
    af_array_bindings(m);
}