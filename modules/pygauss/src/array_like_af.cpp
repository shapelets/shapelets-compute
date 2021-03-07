#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include "pygauss.h"

namespace spd = spdlog;
namespace py = pybind11;

namespace pygauss::arraylike::detail {

    bool is_af_array(const py::object &value) {
        return py::isinstance<af::array>(value) || py::isinstance<ParallelFor>(value);
    }

    std::optional<af::array> from_af_array(const py::object &value) {
        if (py::isinstance<af::array>(value))
            return py::cast<af::array>(value);

        if (py::isinstance<ParallelFor>(value))
            return static_cast<af::array>(py::cast<ParallelFor>(value));

        return std::nullopt;
    }

    inline af::dtype value_or_default(af::dtype _64, af::dtype _32) {
        return af::isDoubleAvailable(af::getDevice()) ? _64 : _32;
    }

    std::optional<af::array> from_scalar(const py::object &value, const af::dim4 &shape) {
        af_array handle = nullptr;
        auto err = AF_SUCCESS;
        if (py::isinstance<py::float_>(value)) {
            auto actual_dtype = value_or_default(af::dtype::f64, af::dtype::f32);
            err = af_constant(&handle, py::cast<double>(value), shape.ndims(), shape.get(), actual_dtype);
        } else if (py::isinstance<py::bool_>(value)) {
            auto v = py::cast<bool>(value) ? 1.0 : 0.0;
            err = af_constant(&handle, v, shape.ndims(), shape.get(), af::dtype::b8);
        } else if (py::isinstance<py::int_>(value)) {
            auto actual_dtype = value_or_default(af::dtype::s64, af::dtype::s32);
            err = af_constant(&handle, (double) py::cast<long>(value), shape.ndims(), shape.get(), actual_dtype);
        } else {
            try {
                auto c = py::cast<std::complex<double>>(value);
                auto actual_dtype = value_or_default(af::dtype::c64, af::dtype::c32);
                af_constant_complex(&handle, c.real(), c.imag(), shape.ndims(), shape.get(), actual_dtype);
            }
            catch (...) {
                err = af_err::AF_ERR_TYPE;
                spd::warn("Expecting a numerical constant but got {}", py::str(value).cast<std::string>());
            }
        }

        if (err != AF_SUCCESS) {
            warn_if_error(err);
            return std::nullopt;
        }
        return af::array(handle);
    }
}
