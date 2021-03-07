#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include <pygauss.h>

namespace spd = spdlog;
namespace py = pybind11;

using dtype_converter = py::detail::type_caster<af::dtype>;

namespace pygauss::arraylike {

    py::buffer_info buffer_protocol(const af::array &self) {

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

        auto isDirect = af::getActiveBackend() != af::Backend::AF_BACKEND_OPENCL;
        void *data = nullptr;

        switch (self.type()) {
            case af::dtype::b8:
            case af::dtype::u8:
                data = isDirect ? self.device<uint8_t>() : self.host<uint8_t>();
                break;
            case af::dtype::s16:
                data = isDirect ? self.device<int16_t>() : self.host<int16_t>();
                break;
            case af::dtype::u16:
                data = isDirect ? self.device<uint16_t>() : self.host<uint16_t>();
                break;
            case af::dtype::f16:
                data = isDirect ? self.device<uint8_t>() : self.host<uint8_t>();
                break;
            case af::dtype::f32:
                data = isDirect ? self.device<float>() : self.host<float>();
                break;
            case af::dtype::s32:
                data = isDirect ? self.device<int32_t>() : self.host<int32_t>();
                break;
            case af::dtype::u32:
                data = isDirect ? self.device<uint32_t>() : self.host<uint32_t>();
                break;
            case af::dtype::u64:
                data = isDirect ? self.device<uint64_t>() : self.host<uint64_t>();
                break;
            case af::dtype::s64:
                data = isDirect ? self.device<int64_t>() : self.host<int64_t>();
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
        /**
         * Checks if an object can be interpreted as a numpy arr_like
         */
        bool is_numpy(const py::object &value) {
            // Try to access the data through py::array...
            auto py_array = py::array::ensure(value);
            if (!py_array)
                return false;

            // Ensure we are not reading a scalar through this method
            // by checking at least one dimension is different to one

            auto i = 0;
            auto ndim = py_array.ndim();
            auto implied_shape = py_array.shape();
            auto all_ones = true;
            while (i < ndim && all_ones) {
                all_ones = (implied_shape[i] == 1);
                i += 1;
            }

            // false if all_ones == true.
            return !all_ones;
        }

        std::optional<af::array> from_numpy(const py::object &value) {
            auto tmp_array = py::array::ensure(value);
            if (!tmp_array)
                return std::nullopt;

            af::dim4 dims(1, 1, 1, 1);
            auto i = 0;
            auto ndim = tmp_array.ndim();
            auto implied_shape = tmp_array.shape();
            while (i < ndim) {
                dims[i] = implied_shape[i];
                i += 1;
            }

            spd::debug("Dims are {}", dims);
            auto arr_type = py::cast<af::dtype>(tmp_array.dtype());
            spd::debug("ArrType is {}", arr_type);

            if (!af::isDoubleAvailable(af::getDevice())) {
                if (arr_type == af::dtype::f64)
                    arr_type = af::dtype::f32;
                else if (arr_type == af::dtype::c64) {
                    arr_type = af::dtype::c32;
                }
            } else if (!af::isHalfAvailable(af::getDevice())) {
                if (arr_type == af::dtype::f16)
                    arr_type = af::dtype::f32;
            }
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
                    auto arr = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<float *>(req.ptr));
                    break;
                }
                case af::dtype::c32: {
                    auto arr = py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast>::ensure(
                            tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<af::cfloat *>(req.ptr));
                    break;
                }
                case af::dtype::f64: {
                    auto arr = py::array_t<double, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<double *>(req.ptr));
                    break;
                }
                case af::dtype::c64: {
                    auto arr = py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast>::ensure(
                            tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<af::cdouble *>(req.ptr));
                    break;
                }
                case af::dtype::b8: {
                    auto arr = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    // missing symbols if bool is used as type param
                    result = af::array(dims, static_cast<uint8_t *>(req.ptr)).as(af::dtype::b8);
                    break;
                }
                case af::dtype::s32: {
                    auto arr = py::array_t<int32_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<int32_t *>(req.ptr));
                    break;
                }
                case af::dtype::u32: {
                    auto arr = py::array_t<uint32_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<uint32_t *>(req.ptr));
                    break;
                }
                case af::dtype::u8: {
                    auto arr = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<uint8_t *>(req.ptr));
                    break;
                }
                case af::dtype::s64: {
                    auto arr = py::array_t<int64_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<int64_t *>(req.ptr));
                    break;
                }
                case af::dtype::u64: {
                    auto arr = py::array_t<uint64_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<uint64_t *>(req.ptr));
                    break;
                }
                case af::dtype::s16: {
                    auto arr = py::array_t<int16_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<int16_t *>(req.ptr));
                    break;
                }
                case af::dtype::u16: {
                    auto arr = py::array_t<uint16_t, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<uint16_t *>(req.ptr));
                    break;
                }
                case af::dtype::f16: {
                    if (!af::isHalfAvailable(af::getDevice()))
                        throw std::runtime_error("Current device doesn't support half floats");

                    // there is no way around at the moment.
                    auto arr = py::array_t<float, py::array::f_style | py::array::forcecast>::ensure(tmp_array);
                    auto req = arr.request();
                    result = af::array(dims, static_cast<float *>(req.ptr)).as(af::dtype::f16);
                    break;
                }
                default: {
                    spdlog::warn("Unsupported data type {}", arr_type);
                    return std::nullopt;
                }

            }
            return result;
        }
    }
}
