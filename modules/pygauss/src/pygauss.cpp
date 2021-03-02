
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include "pygauss.h"

namespace py = pybind11;
namespace spd = spdlog;

PYBIND11_MODULE(pygauss, m) {
    m.doc() = R"(
        Khiva module
        ------------

        .. currentmodule:: khiva

        .. autosummary::
            :toctree: _generate
        
    )";
    spd::set_level(spd::level::level_enum::debug);
#ifndef NDEBUG
    spd::set_level(spd::level::level_enum::debug);
#endif
    af_array_bindings(m);
    gauss_matrix_bindings(m);
}
