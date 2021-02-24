#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <af_array/af_array.h>

py::dtype to_python(const af::dtype &value) {
    switch (value) {
        case af::dtype::b8:
            return py::dtype::of<bool>();
        case af::dtype::u8:
            return py::dtype::of<uint8_t>();
        case af::dtype::s16:
            return py::dtype::of<int16_t>();
        case af::dtype::u16:
            return py::dtype::of<uint16_t>();
        case af::dtype::f16:
            return py::dtype("float16");
        case af::dtype::f32:
            return py::dtype::of<float>();
        case af::dtype::s32:
            return py::dtype::of<int32_t>();
        case af::dtype::u32:
            return py::dtype::of<uint32_t>();
        case af::dtype::u64:
            return py::dtype::of<uint64_t>();
        case af::dtype::s64:
            return py::dtype::of<int64_t>();
        case af::dtype::f64:
            return py::dtype::of<double>();
        case af::dtype::c32:
            return py::dtype::of<std::complex<float>>();
        case af::dtype::c64:
            return py::dtype::of<std::complex<double>>();
    }
}

std::optional<af::dim4> to_af_dim4(const py::handle &src) {

    std::optional<af::dim4> result;

    if (src.ptr() != nullptr && !src.is_none()) {

        if (py::isinstance<py::int_>(src)) {
            result = af::dim4(py::cast<long>(src));
        }

        if (py::isinstance<py::tuple>(src)) {
            auto tuple = py::cast<py::tuple>(src);
            auto dms = af::dim4(1, 1, 1, 1);
            int i = 0;
            for (auto const &d: tuple) {
                dms[i++] = py::cast<long>(d);
            }
            result = dms;
        }

    }

    return result;
}

af::dtype to_af_dtype(const py::handle &src) {

    if (src.ptr() == nullptr || src.is_none() || !py::isinstance<py::object>(src))
        throw std::runtime_error("Invalid handle for dtype conversion.");

    auto asObject = py::cast<py::object>(src);
    auto type = py::dtype::from_args(std::move(asObject));

    auto size = type.itemsize();
    switch (type.kind()) {
        case 'b':
            return af::dtype::b8;
        case 'i':
            switch (size) {
                case 1:
                    return af::dtype::u8;
                case 2:
                    return af::dtype::s16;
                case 4:
                    return af::dtype::s32;
                case 8:
                    return af::dtype::s64;
                default:
                    throw std::runtime_error("Unsupported signed type.");
            }
        case 'u':
            switch (size) {
                case 1:
                    return af::dtype::u8;
                case 2:
                    return af::dtype::u16;
                case 4:
                    return af::dtype::u32;
                case 8:
                    return af::dtype::u64;
                default:
                    throw std::runtime_error("Unsupported unsigned type.");
            }
        case 'f':
            switch (size) {
                case 2:
                    return af::dtype::f16;
                case 4:
                    return af::dtype::f32;
                case 8:
                    return af::dtype::f64;
                default:
                    throw std::runtime_error("Unsupported float type.");
            }
        case 'c':
            switch (size) {
                case 8:
                    return af::dtype::c32;
                case 16:
                    return af::dtype::c64;
                default:
                    throw std::runtime_error("Unsupported complex type.");
            }
        default:
            throw std::runtime_error("Unsupported data type.");
    }
}

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

af::dtype value_or_default(const std::optional<af::dtype> &type, af::dtype _64, af::dtype _32) {
    return type.value_or(af::isDoubleAvailable(af::getDevice()) ? _64 : _32);
}

af_array constant_array(const py::object &value, const af::dim4 &shape, const std::optional<af::dtype> &type) {
    af_array handle = nullptr;

    if (py::isinstance<py::bool_>(value)) {
        auto v = py::cast<bool>(value) ? 1.0 : 0.0;
        check_af_error(af_constant(&handle, v, shape.ndims(), shape.get(), type.value_or(af::dtype::b8)));
    } else if (py::isinstance<py::int_>(value)) {
        auto actual_dtype = value_or_default(type, af::dtype::s64, af::dtype::s32);
        check_af_error(af_constant(&handle, (double) py::cast<long>(value), shape.ndims(), shape.get(), actual_dtype));
    } else if (py::isinstance<py::float_>(value)) {
        auto actual_dtype = value_or_default(type, af::dtype::f64, af::dtype::f32);
        check_af_error(af_constant(&handle, py::cast<double>(value), shape.ndims(), shape.get(), actual_dtype));
    } else {
        try {
            auto c = py::cast<std::complex<double>>(value);
            auto actual_dtype = value_or_default(type, af::dtype::c64, af::dtype::c32);
            check_af_error(af_constant_complex(&handle, c.real(), c.imag(), shape.ndims(), shape.get(), actual_dtype));
        }
        catch (...) {
            throw std::runtime_error(
                    "Expecting a numerical constant but got [" + py::str(value).cast<std::string>() + "]");
        }
    }

    return handle;
}
