#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include "pygauss.h"

namespace spd = spdlog;
namespace py = pybind11;

namespace pygauss::arraylike::detail {

    bool af_is_array(const py::object &value) {
        return py::isinstance<af::array>(value) || py::isinstance<ParallelFor>(value);
    }

    af::array from_af_array(const py::object &value) {
        if (py::isinstance<af::array>(value))
            return py::cast<af::array>(value);

        if (py::isinstance<ParallelFor>(value))
            return static_cast<af::array>(py::cast<ParallelFor>(value));

        throw std::invalid_argument("Value not a valid ShapeletsArray object");
    }
}
