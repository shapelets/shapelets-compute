#ifndef __TYPECASTERS_H__
#define __TYPECASTERS_H__


#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>

#include "formatters.h"

namespace py = pybind11;
namespace spd = spdlog;

using half = half_float::half;

namespace pybind11::detail {

    constexpr int NPY_FLOAT16 = 23;

    template <>
    struct npy_format_descriptor<half> {
        static pybind11::dtype dtype() {
            handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
            return reinterpret_borrow<pybind11::dtype>(ptr);
        }
        static std::string format() {
            // following: https://docs.python.org/3/library/struct.html#format-characters
            return "e";
        }
        static constexpr auto name() {
            return _("float16");
        }
    };

//    template <>
//    struct type_caster<half> : npy_scalar_caster<half> {
//        static constexpr auto name = _("float16");
//    };


    template<>
    struct type_caster<af::dtype> {

        bool load(handle src, bool) {
            auto converted = load(src);
            if (converted.has_value()) {
                value = converted.value();
                return true;
            }
            return false;
        }

        static handle cast(af::dtype type, return_value_policy /*policy*/, handle /*parent*/) {
            return cast(type).release();
        }

    PYBIND11_TYPE_CASTER(af::dtype, _("DataType"));

    public:
        static std::optional<af::dtype> load(const pybind11::handle &src) {

            if (src.ptr() == nullptr || src.is_none() || !pybind11::isinstance<pybind11::object>(src))
                return std::nullopt;

            auto asObject = pybind11::cast<pybind11::object>(src);
            auto type = pybind11::dtype::from_args(std::move(asObject));

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
                            return std::nullopt;
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
                            return std::nullopt;
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
                            return std::nullopt;
                    }
                case 'c':
                    switch (size) {
                        case 8:
                            return af::dtype::c32;
                        case 16:
                            return af::dtype::c64;
                        default:
                            return std::nullopt;
                    }
                default:
                    return std::nullopt;
            }
        }

        static pybind11::dtype cast(const af::dtype &value) {
            switch (value) {
                case af::dtype::b8:
                    return pybind11::dtype::of<bool>();
                case af::dtype::u8:
                    return pybind11::dtype::of<uint8_t>();
                case af::dtype::s16:
                    return pybind11::dtype::of<int16_t>();
                case af::dtype::u16:
                    return pybind11::dtype::of<uint16_t>();
                case af::dtype::f16:
                    return pybind11::dtype("float16");
                case af::dtype::f32:
                    return pybind11::dtype::of<float>();
                case af::dtype::s32:
                    return pybind11::dtype::of<int32_t>();
                case af::dtype::u32:
                    return pybind11::dtype::of<uint32_t>();
                case af::dtype::u64:
                    return pybind11::dtype::of<uint64_t>();
                case af::dtype::s64:
                    return pybind11::dtype::of<int64_t>();
                case af::dtype::f64:
                    return pybind11::dtype::of<double>();
                case af::dtype::c32:
                    return pybind11::dtype::of<std::complex<float>>();
                case af::dtype::c64:
                default:
                    return pybind11::dtype::of<std::complex<double>>();
            }
        }
    };

    template<>
    struct type_caster<af::dim4> {
        bool load(handle src, bool) {
            auto converted = load(src);
            if (converted.has_value()) {
                value = converted.value();
                return true;
            }
            return false;
        }

        static handle cast(af::dim4 dms, return_value_policy /*policy*/, handle /*parent*/) {
            return cast(dms).release();
        }

    PYBIND11_TYPE_CASTER(af::dim4, _("Shape"));

    public:
        static std::optional<af::dim4> load(const pybind11::handle &src) {

            std::optional<af::dim4> result;

            if (src.ptr() != nullptr && !src.is_none()) {

                if (pybind11::isinstance<pybind11::int_>(src)) {
                    result = af::dim4(pybind11::cast<long>(src));
                }

                if (pybind11::isinstance<pybind11::tuple>(src)) {
                    auto tuple = pybind11::cast<pybind11::tuple>(src);
                    auto dms = af::dim4(1, 1, 1, 1);
                    int i = 0;
                    for (auto const &d: tuple) {
                        dms[i++] = pybind11::cast<dim_t>(d);
                    }
                    result = dms;
                }

            }

            return result;
        }

        static pybind11::tuple cast(af::dim4 dms) {
            switch (dms.ndims()) {
                case 1:
                    // this is to avoid problems with matplotlib!! Duh!!
                    return pybind11::make_tuple(dms[0], 1);
                case 2:
                    return pybind11::make_tuple(dms[0], dms[1]);
                case 3:
                    return pybind11::make_tuple(dms[0], dms[1], dms[2]);
                default:
                    return pybind11::make_tuple(dms[0], dms[1], dms[2], dms[3]);
            }
        }
    };
}

#endif  // __TYPECASTERS_H__
