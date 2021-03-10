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

    af::dtype harmonize_types(const af::dtype scalar_type, const af::dtype array_type) {
        // If same, do not do anything
        if (scalar_type == array_type) { return scalar_type; }

        // If complex, return appropriate complex type
        if (scalar_type == c32 || scalar_type == c64) {
            if (array_type == f64 || array_type == c64)
                return c64;
            return c32;
        }

        // If 64 bit precision, do not lose precision
        if (array_type == f64 || array_type == c64 || array_type == f32 || array_type == c32) {
            return array_type;
        }

        // If the array is f16 then avoid upcasting to float or double
        if ((scalar_type == f64 || scalar_type == f32) && (array_type == f16)) {
            return f16;
        }

        // Default to single precision by default when multiplying with scalar
        if (scalar_type == f64) {
            return f32;
        }

        // Punt to C api for everything else
        return scalar_type;
    }
}
