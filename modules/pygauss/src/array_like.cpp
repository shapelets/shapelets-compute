

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pygauss.h>

namespace py = pybind11;

namespace pygauss::arraylike {

    int scalar_size(const af::dtype &dtype) {
        switch (dtype) {
            case af::dtype::b8:
            case af::dtype::u8:
                return 1;
            case af::dtype::s16:
            case af::dtype::u16:
            case af::dtype::f16:
                return 2;
            case af::dtype::f32:
            case af::dtype::s32:
            case af::dtype::u32:
                return 4;
            case af::dtype::u64:
            case af::dtype::s64:
            case af::dtype::f64:
            case af::dtype::c32:
                return 8;
            case af::dtype::c64:
                return 16;
        }
    }

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

    std::optional<af::array> try_cast(const py::object &value,
                                      const std::optional<af::dim4> &shape,
                                      const std::optional<af::dtype> &dtype) {

        std::optional<af::array> result = std::nullopt;
        if (detail::is_af_array(value)) {
            spd::debug("is_af_array OK");
            result = detail::from_af_array(value);
        } else if (detail::is_arrow(value)) {
            spd::debug("is_arrow OK");
            result = detail::from_arrow(value);
        } else if (detail::is_numpy(value)) {
            spd::debug("is_numpy OK");
            result = detail::from_numpy(value);
        } else if (is_scalar(value) && shape.has_value()) {
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

    af::array cast(const py::object &array_like,
                   const std::optional<af::dim4> &shape,
                   const std::optional<af::dtype> &dtype) {

        auto a = try_cast(array_like, shape, dtype);
        if (a.has_value())
            return a.value();

        std::ostringstream stm;
        stm << "Unable to process " << py::repr(array_like) << "as a valid tensor.";
        auto err_msg = stm.str();
        spd::error(err_msg);
        throw std::runtime_error(err_msg);
    }
}
