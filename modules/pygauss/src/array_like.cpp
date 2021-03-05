

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "af_array/af_array.h"
#include "array_like.h"

namespace py = pybind11;

namespace array_like {

    bool is_scalar(const py::object &value) {
        return py::isinstance<py::float_>(value)
               || py::isinstance<py::int_>(value)
               || py::isinstance<py::bool_>(value)
               || py::isinstance<std::complex<double>>(value);
    }

    bool is_array(const py::object &value) {
        return detail::is_af_array(value)
        || detail::is_arrow(value)
        || detail::is_numpy(value);
    }

    std::optional<af::array> to_array(const py::object &value,
                                      const std::optional<af::dim4> &shape,
                                      const std::optional<af::dtype> &dtype) {

        std::optional<af::array> result = std::nullopt;
        if (detail::is_af_array(value)) {
            spd::debug("is_af_array OK");
            result = detail::from_af_array(value);
        }
        else if (detail::is_arrow(value)) {
            spd::debug("is_arrow OK");
            result = detail::from_arrow(value);
        }
        else if (detail::is_numpy(value)) {
            spd::debug("is_numpy OK");
            result = detail::from_numpy(value);
        }
        else if (is_scalar(value) && shape.has_value()) {
            spd::debug("is_scalar and shape OK");
            result = detail::from_scalar(value, shape.value());
        }

        if (!result.has_value())
            return result;

        // Adjust to spec
        auto candidate = result.value();
        if (shape.has_value()) {
            auto given_shape = shape.value();
            if (candidate.dims() != given_shape && candidate.elements() == given_shape.elements()) {
                candidate = af::moddims(candidate, given_shape);
            } else {
                std::stringstream err_msg;
                err_msg << "Unable to change tensor geometry from " << candidate.dims() << " to " << given_shape;
                throw std::runtime_error(err_msg.str());
            }
        }

        if (dtype.has_value()) {
            auto given_type = dtype.value();
            if (candidate.type() != given_type) {
                candidate = candidate.as(given_type);
            }
        }

        return candidate;
    }
}
