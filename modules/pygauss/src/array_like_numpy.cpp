/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <optional>
#include <pygauss.h>

namespace spd = spdlog;
namespace py = pybind11;

using dtype_converter = py::detail::type_caster<af::dtype>;
using half_float::half;

namespace pygauss::arraylike {

    py::buffer_info numpy_buffer_protocol(const af::array &self) {

        // Ensure all the computations have completed.
        af::eval(self);

        // get the type
        auto fmt = dtype_converter::cast(self.type());

        // get the scalar size
        ssize_t scalarSize = fmt.itemsize();

        // get the number of dimensions
        auto dimCount = self.numdims();

        // get the shape
        auto shape = self.dims();

        // convert shape
        std::vector<ssize_t> dimensions(dimCount);
        for (size_t i = 0; i < dimCount; i++) {
            dimensions[i] = shape[i];
        }

        // create strides
        std::vector<ssize_t> strides(dimCount, scalarSize);
        for (size_t i = 1; i < dimCount; ++i)
            strides[i] = strides[i - 1] * shape[i - 1];

        af_backend arr_backend = AF_BACKEND_DEFAULT;
        af_get_backend_id(&arr_backend, self.get());
        auto isDirect = arr_backend == AF_BACKEND_CPU;

        void *data = nullptr;

        switch (self.type()) {
            case af::dtype::b8:
            case af::dtype::u8:
                data = isDirect ? self.device<unsigned char>() : self.host<unsigned char>();
                break;
            case af::dtype::s16:
                data = isDirect ? self.device<short>() : self.host<short>();
                break;
            case af::dtype::u16:
                data = isDirect ? self.device<unsigned short>() : self.host<unsigned short>();
                break;
            case af::dtype::f16:
                data = isDirect ? self.device<af::half>() : self.host<af::half>();
                break;
            case af::dtype::f32:
                data = isDirect ? self.device<float>() : self.host<float>();
                break;
            case af::dtype::s32:
                data = isDirect ? self.device<int>() : self.host<int>();
                break;
            case af::dtype::u32:
                data = isDirect ? self.device<unsigned int>() : self.host<unsigned int>();
                break;
            case af::dtype::u64:
                data = isDirect ? self.device<unsigned long long>() : self.host<unsigned long long>();
                break;
            case af::dtype::s64:
                data = isDirect ? self.device<long long>() : self.host<long long>();
                break;
            case af::dtype::f64:
                data = isDirect ? self.device<double>() : self.host<double>();
                break;
            case af::dtype::c32:
                data = isDirect ? self.device<af::cfloat>() : self.host<af::cfloat>();
                break;
            case af::dtype::c64:
                data = isDirect ? self.device<af::cdouble>() : self.host<af::cdouble>();
                break;
        }

        auto capsule = isDirect ?
                       py::capsule(&self, [](void *f) {
                           spd::debug("Unlocking array");
                           static_cast<af::array *>(f)->unlock();
                       }) :
                       py::capsule(data, [](void *f) {
                           spd::debug("Freeing host pointer");
                           af::freeHost(f);
                       });

        spd::debug("Returning a {}writable of type [{}-{}] {}",
                   isDirect ? "" : "non ",
                   fmt.kind(),
                   fmt.itemsize(),
                   shape);

        return py::array(fmt, dimensions, strides, data, capsule).request(isDirect);
    }

    namespace detail {

        bool numpy_is_array(const py::object &value) {
            // Can be understood as an array?
            auto arr = py::array::ensure(value);
            if (!arr) return false;

            // Got dimensions?
            auto has_dimensions = arr.ndim() > 0 && arr.size() > 1;
            if (!has_dimensions) return false;

            // supported array fire type?
            auto type = dtype_converter::load(arr.dtype().cast<py::handle>());
            return type.has_value();
        }

        bool numpy_is_scalar(const py::object &value) {
            // Can be understood and converted...
            auto arr = py::array::ensure(value);
            if (!arr) return false;

            // Should not have dimensions
            auto has_dimensions = arr.ndim() > 0 && arr.size() > 1;
            if (has_dimensions) return false;

            // is supported by an arrayfire type?
            auto type = dtype_converter::load(arr.dtype().cast<py::handle>());
            return type.has_value();
        }

        inline af::dtype choose_type(af::dtype _64, af::dtype _32) {
            return af::isDoubleAvailable(af::getDevice()) ? _64 : _32;
        }

        af::array numpy_scalar_to_array(const py::object &value, const af::dim4 &shape, const af::dtype& ref_type) {
            auto arr = py::array::ensure(value);
            if (!arr) throw std::invalid_argument("Value not a valid numpy object");

            auto has_dimensions = arr.ndim() > 0 && arr.size() > 1;
            if (has_dimensions) throw std::invalid_argument("Value is not a numpy scalar");

            auto type = dtype_converter::load(arr.dtype().cast<py::handle>());
            if (!type.has_value()) throw std::invalid_argument("Unsupported value type.");

            auto actual_type = harmonize_types(type.value(), ref_type);

            af_array handle = nullptr;
            af_err err;
            switch (type.value()) {
                case af::dtype::c32: {
                    auto ptr = arr.unchecked<std::complex<float>>()(0);
                    auto real = static_cast<double>(ptr.real());
                    auto imag = static_cast<double>(ptr.imag());
                    err = af_constant_complex(&handle, real, imag, shape.ndims(), shape.get(), actual_type);
                    break;
                }
                case af::dtype::c64: {
                    auto ptr = arr.unchecked<std::complex<double>>()(0);
                    err = af_constant_complex(&handle, ptr.real(), ptr.imag(), shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::s16: {
                    auto data = arr.unchecked<int16_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }
                case af::dtype::s32: {
                    auto data = arr.unchecked<int32_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }
                case af::dtype::s64: {
                    auto data = arr.unchecked<int64_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::f16: {
                    auto data = arr.unchecked<half>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::f32: {
                    auto data = arr.unchecked<float>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::f64: {
                    auto data = arr.unchecked<double>()(0);
                    err = af_constant(&handle, data, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::u8: {
                    auto data = arr.unchecked<uint8_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::u16: {
                    auto data = arr.unchecked<uint16_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::u32: {
                    auto data = arr.unchecked<uint32_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }

                case af::dtype::u64: {
                    auto data = arr.unchecked<uint64_t>()(0);
                    auto casted = static_cast<double>(data);
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(),actual_type);
                    break;
                }

                case af::dtype::b8: {
                    auto data = arr.unchecked<uint8_t>()(0);
                    auto casted = data == 1 ? 1.0 : 0.0;
                    err = af_constant(&handle, casted, shape.ndims(), shape.get(), actual_type);
                    break;
                }
            }
            // Throw if the operation failed to convert
            throw_on_error(err);
            // otherwise, return the new array.
            return af::array(handle);
        }


        af::array numpy_array_to_array(const py::object &value) {
            auto np_arr = py::array::ensure(value);
            if (!np_arr) throw std::invalid_argument("Value not a valid numpy object");

            af::dim4 dims(1, 1, 1, 1);
            auto i = 0;
            auto ndim = np_arr.ndim();
            auto implied_shape = np_arr.shape();
            while (i < ndim) {
                dims[i] = implied_shape[i];
                i += 1;
            }

            spd::debug("Dims are {}", dims);
            auto arr_type = py::cast<af::dtype>(np_arr.dtype());
            spd::debug("ArrType is {}", arr_type);

            auto isDblAvailable = af::isDoubleAvailable(af::getDevice());
            auto isHalfAvailable = af::isHalfAvailable(af::getDevice());
            if (arr_type == af::dtype::f64 && !isDblAvailable)
                arr_type = af::dtype::f32;
            if (arr_type == af::dtype::c64 && !isDblAvailable)
                arr_type = af::dtype::c32;
            if (arr_type == af::dtype::f16 && !isHalfAvailable)
                arr_type = af::dtype::f32;

            spd::debug("Final arrType is {}", arr_type);

            /*
             * TODO
             *
             * At the moment the code below doesn't check for:
             * a) if the data is a vector, it is not necessary to request a f_style
             * b) the data taken from host memory, which implies always a copy; for cpu and cuda
             *    devices, we may be able to just put the data in the devices memory.
             */
            af::array result;
            switch (arr_type) {
                case af::dtype::f32: {
                    auto arr = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<float *>(req.ptr));
                }
                case af::dtype::c32: {
                    auto arr = py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast>::ensure(
                            np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<af::cfloat *>(req.ptr));
                }
                case af::dtype::f64: {
                    auto arr = py::array_t<double, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<double *>(req.ptr));
                }
                case af::dtype::c64: {
                    auto arr = py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast>::ensure(
                            np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<af::cdouble *>(req.ptr));
                }
                case af::dtype::b8: {
                    auto arr = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    // missing symbols if bool is used as type param
                    return af::array(dims, static_cast<unsigned char *>(req.ptr)).as(af::dtype::b8);
                }
                case af::dtype::s32: {
                    auto arr = py::array_t<int32_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<int *>(req.ptr));
                }
                case af::dtype::u32: {
                    auto arr = py::array_t<uint32_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<unsigned int *>(req.ptr));
                }
                case af::dtype::u8: {
                    auto arr = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<unsigned char *>(req.ptr));
                }
                case af::dtype::s64: {
                    auto arr = py::array_t<int64_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<long long *>(req.ptr));
                }
                case af::dtype::u64: {
                    auto arr = py::array_t<uint64_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<unsigned long long *>(req.ptr));
                }
                case af::dtype::s16: {
                    auto arr = py::array_t<int16_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<short *>(req.ptr));
                }
                case af::dtype::u16: {
                    auto arr = py::array_t<uint16_t, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    return af::array(dims, static_cast<unsigned short *>(req.ptr));
                }
                case af::dtype::f16: {
                    auto arr = py::array_t<half, py::array::f_style | py::array::forcecast>::ensure(np_arr);
                    auto req = arr.request();
                    // note we are using af::half
                    return af::array(dims, static_cast<af::half *>(req.ptr));
                }
                default: {
                    std::ostringstream msg;
                    msg << "Unknown data type [" << arr_type << "]";
                    throw std::runtime_error(msg.str());
                }
            }
        }
    }
}
