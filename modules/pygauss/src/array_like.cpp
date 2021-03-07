

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

        if (value.is_none())
            return std::nullopt;

        if (detail::is_af_array(value)) {
            spd::debug("is_af_array OK");
            return detail::from_af_array(value);
        }

        if (detail::is_arrow(value)) {
            spd::debug("is_arrow OK");
            return detail::from_arrow(value);
        }

        if (detail::is_numpy(value)) {
            spd::debug("is_numpy OK");
            return detail::from_numpy(value);
        }

        if (is_scalar(value) && shape.has_value()) {
            spd::debug("is_scalar and shape OK");
            return detail::from_scalar(value, shape.value(), dtype);
        }

        return std::nullopt;
    }

    af::array cast_and_adjust(const py::object &array_like,
                              const std::optional<af::dim4> &shape,
                              const std::optional<af::dtype> &dtype) {
        auto a = try_cast(array_like, shape, dtype);
        if (!a.has_value()) {
            std::ostringstream stm;
            stm << "Unable to process " << py::repr(array_like) << "as a valid tensor.";
            auto err_msg = stm.str();
            spd::error(err_msg);
            throw std::runtime_error(err_msg);
        }

        auto candidate = a.value();
        if (dtype.has_value()) {
            if (candidate.type() != dtype.value()) {
                candidate = candidate.as(dtype.value());
            }
        }

        if (shape.has_value()) {
            if (candidate.dims() != shape.value()) {
                if (candidate.elements() == shape.value().elements()) {
                    candidate = af::moddims(candidate, shape.value());
                }
                else {
                    std::ostringstream stm;
                    stm << "Unable to adjust shape of tensor since the number of elements are different. ";
                    stm << "Source has " << candidate.elements() << " and target requires " << shape.value().elements();
                    auto err_msg = stm.str();
                    spd::error(err_msg);
                    throw std::runtime_error(err_msg);
                }
            }
        }

        return candidate;
    }

    af::array cast(const py::object &array_like,
                   bool floating,
                   const std::optional<af::dim4> &shape,
                   const std::optional<af::dtype> &dtype) {

        auto a = try_cast(array_like, shape, dtype);
        if (a.has_value()) {
            auto result = a.value();
            if (floating)
                result = ensure_floating(result);
            return result;
        }

        std::ostringstream stm;
        stm << "Unable to process " << py::repr(array_like) << "as a valid tensor.";
        auto err_msg = stm.str();
        spd::error(err_msg);
        throw std::runtime_error(err_msg);
    }

    af::array ensure_floating(const af::array& src) {
        if (src.isfloating())
            return src;

        auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
        PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
        return src.as(safe_conversion);
    }
}
