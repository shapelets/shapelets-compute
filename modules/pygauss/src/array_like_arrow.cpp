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
