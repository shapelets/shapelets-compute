#ifndef AF_ARRAY_PYBIND11_CONVERSIONS__H
#define AF_ARRAY_PYBIND11_CONVERSIONS__H

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fmt/format.h>

template <>
struct fmt::formatter<af::dim4> {
    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }
    template <typename FormatContext>
    auto format(const af::dim4& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}]", p[0], p[1],p[2],p[3]);
    }
};

template <>
struct fmt::formatter<af::dtype> {
    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }
    template <typename FormatContext>
    auto format(const af::dtype& p, FormatContext& ctx) {
        string_view name = "unknown";
        switch (p) {
            case af::dtype::b8: name = "b8"; break;
            case af::dtype::f32: name = "f32"; break;
            case af::dtype::c64: name = "c64"; break;
            case af::dtype::f64: name = "f64"; break;
            case af::dtype::f16: name = "f16"; break;
            case af::dtype::c32: name = "c32"; break;
            case af::dtype::s16: name = "s16"; break;
            case af::dtype::u8: name = "u8"; break;
            case af::dtype::s32: name = "s32"; break;
            case af::dtype::s64: name = "s64"; break;
            case af::dtype::u16: name = "u16"; break;
            case af::dtype::u32: name = "u32"; break;
            case af::dtype::u64: name = "u64"; break;
        }

        return format_to(ctx.out(), "(af::dtype) {}", name);
    }
};

template <>
struct fmt::formatter<af_err> {
    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }
    template <typename FormatContext>
    auto format(const af_err& p, FormatContext& ctx) {
        string_view name = "unknown";
        switch (p) {
            case af_err::AF_SUCCESS: name = "AF_SUCCESS"; break;
            case af_err::AF_ERR_ARG: name = "AF_ERR_ARG"; break;
            case af_err::AF_ERR_ARR_BKND_MISMATCH: name = "AF_ERR_ARR_BKND_MISMATCH"; break;
            case af_err::AF_ERR_BATCH: name = "AF_ERR_BATCH"; break;
            case af_err::AF_ERR_DEVICE: name = "AF_ERR_DEVICE"; break;
            case af_err::AF_ERR_DIFF_TYPE: name = "AF_ERR_DIFF_TYPE"; break;
            case af_err::AF_ERR_DRIVER: name = "AF_ERR_DRIVER"; break;
            case af_err::AF_ERR_INTERNAL: name = "AF_ERR_INTERNAL"; break;
            case af_err::AF_ERR_INVALID_ARRAY: name = "AF_ERR_INVALID_ARRAY"; break;
            case af_err::AF_ERR_LOAD_LIB: name = "AF_ERR_LOAD_LIB"; break;
            case af_err::AF_ERR_LOAD_SYM: name = "AF_ERR_LOAD_SYM"; break;
            case af_err::AF_ERR_NONFREE: name = "AF_ERR_NONFREE"; break;
            case af_err::AF_ERR_NOT_CONFIGURED: name = "AF_ERR_NOT_CONFIGURED"; break;
            case af_err::AF_ERR_NOT_SUPPORTED: name = "AF_ERR_NOT_SUPPORTED"; break;
            case af_err::AF_ERR_NO_DBL: name = "AF_ERR_NO_DBL"; break;
            case af_err::AF_ERR_NO_GFX: name = "AF_ERR_NO_GFX"; break;
            case af_err::AF_ERR_NO_HALF: name = "AF_ERR_NO_HALF"; break;
            case af_err::AF_ERR_NO_MEM: name = "AF_ERR_NO_MEM"; break;
            case af_err::AF_ERR_RUNTIME: name = "AF_ERR_RUNTIME"; break;
            case af_err::AF_ERR_SIZE: name = "AF_ERR_SIZE"; break;
            case af_err::AF_ERR_TYPE: name = "AF_ERR_TYPE"; break;
            case af_err::AF_ERR_UNKNOWN: name = "AF_ERR_UNKNOWN"; break;
        }

        return format_to(ctx.out(), "(af_err) {} {}", (long long)p, name);
    }
};

pybind11::dtype to_python(const af::dtype &value);
std::optional<af::dim4> to_af_dim4(const pybind11::handle& src);
af::dtype to_af_dtype(const pybind11::handle& src);
int scalar_size(const af::dtype& dtype);
af_array constant_array(const pybind11::object& src, const af::dim4 &shape, const std::optional<af::dtype> &type = std::nullopt);

namespace pybind11::detail {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"

    template<>
    struct type_caster<af::dtype> {

        bool load(handle src, bool) {
            try {
                value = to_af_dtype(src);
                return true;
            }
            catch (...) {
                return false;
            }
        }

        static handle cast(af::dtype type, return_value_policy /*policy*/, handle /*parent*/) {
            return to_python(type).release();
        }

    PYBIND11_TYPE_CASTER(af::dtype, _("DataType"));
    };

    template<>
    struct type_caster<af::dim4> {
        bool load(handle src, bool) {

            auto converted = to_af_dim4(src);
            if (converted.has_value()) {
                value = converted.value();
                return true;
            }

            return false;
        }

        static handle cast(af::dim4 dms, return_value_policy /*policy*/, handle /*parent*/) {
            switch (dms.ndims()) {
                case 1:
                    // this is to avoid problems with matplotlib!! Duh!!
                    return pybind11::make_tuple(dms[0], 1).release();
                case 2:
                    return pybind11::make_tuple(dms[0], dms[1]).release();
                case 3:
                    return pybind11::make_tuple(dms[0], dms[1], dms[2]).release();
                default:
                    return pybind11::make_tuple(dms[0], dms[1], dms[2], dms[3]).release();
            }
        }

    PYBIND11_TYPE_CASTER(af::dim4, _("Shape"));
    };

#pragma clang diagnostic pop
}

#endif // AF_ARRAY_PYBIND11_CONVERSIONS__H
