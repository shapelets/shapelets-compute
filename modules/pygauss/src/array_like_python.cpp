#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

namespace pygauss::arraylike::detail {

bool python_is_scalar(const py::object& value) {
    return py::isinstance<py::float_>(value)
           || py::isinstance<py::int_>(value)
           || py::isinstance<py::bool_>(value)
           || py::isinstance<std::complex<double>>(value);
}

af::array python_scalar_to_array(const py::object& value, const af::dim4 &shape, const af::dtype& ref_type) {
    af_array handle = nullptr;

    auto err = AF_SUCCESS;
    if (py::isinstance<py::float_>(value)) {
        auto actual_dtype = harmonize_types(af::dtype::f64, ref_type);
        err = af_constant(&handle, py::cast<double>(value), shape.ndims(), shape.get(), actual_dtype);

    } else if (py::isinstance<py::bool_>(value)) {
        auto actual_dtype = harmonize_types(af::dtype::b8, ref_type);
        auto v = py::cast<bool>(value) ? 1.0 : 0.0;
        err = af_constant(&handle, v, shape.ndims(), shape.get(), actual_dtype);

    } else if (py::isinstance<py::int_>(value)) {
        auto actual_dtype = harmonize_types(af::dtype::s64, ref_type);
        err = af_constant(&handle, (double) py::cast<long>(value), shape.ndims(), shape.get(), actual_dtype);

    } else if (py::isinstance<std::complex<double>>(value)) {
        auto c = py::cast<std::complex<double>>(value);
        auto actual_dtype = harmonize_types(af::dtype::c64, ref_type);
        err = af_constant_complex(&handle, c.real(), c.imag(), shape.ndims(), shape.get(), actual_dtype);
    } else {
        // Unsupported
        auto value_as_str = py::repr(value).cast<std::string>();
        auto value_type_str = py::repr(py::type(value)).cast<std::string>();
        std::stringstream buffer;
        buffer << "Expecting a python constant but got " << value_as_str << " ";
        buffer << "of type " << value_type_str;
        throw std::invalid_argument(buffer.str());
    }

    // Throw if the operation failed to convert
    throw_on_error(err);
    // otherwise, return the new array.
    return af::array(handle);
}

}
