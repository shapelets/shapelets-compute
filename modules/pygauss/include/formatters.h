#ifndef __FORMATTERS_H__
#define __FORMATTERS_H__

#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

/**
 * Formatter to log af::dim4 instances
 */
template<>
struct fmt::formatter<af::dim4> {
    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto format(const af::dim4 &p, FormatContext &ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}]", p[0], p[1], p[2], p[3]);
    }
};

/**
 * Formatter to log af::dtype instances
 */
template<>
struct fmt::formatter<af::dtype> {
    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto format(const af::dtype &p, FormatContext &ctx) {
        string_view name = "unknown";
        switch (p) {
            case af::dtype::b8:
                name = "b8";
                break;
            case af::dtype::f32:
                name = "f32";
                break;
            case af::dtype::c64:
                name = "c64";
                break;
            case af::dtype::f64:
                name = "f64";
                break;
            case af::dtype::f16:
                name = "f16";
                break;
            case af::dtype::c32:
                name = "c32";
                break;
            case af::dtype::s16:
                name = "s16";
                break;
            case af::dtype::u8:
                name = "u8";
                break;
            case af::dtype::s32:
                name = "s32";
                break;
            case af::dtype::s64:
                name = "s64";
                break;
            case af::dtype::u16:
                name = "u16";
                break;
            case af::dtype::u32:
                name = "u32";
                break;
            case af::dtype::u64:
                name = "u64";
                break;
        }

        return format_to(ctx.out(), "(af::dtype) {}", name);
    }
};

/**
 * Formatter to log af_err
 */
template<>
struct fmt::formatter<af_err> {
    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto format(const af_err &p, FormatContext &ctx) {
        string_view name = "unknown";
        switch (p) {
            case af_err::AF_SUCCESS:
                name = "AF_SUCCESS";
                break;
            case af_err::AF_ERR_ARG:
                name = "AF_ERR_ARG";
                break;
            case af_err::AF_ERR_ARR_BKND_MISMATCH:
                name = "AF_ERR_ARR_BKND_MISMATCH";
                break;
            case af_err::AF_ERR_BATCH:
                name = "AF_ERR_BATCH";
                break;
            case af_err::AF_ERR_DEVICE:
                name = "AF_ERR_DEVICE";
                break;
            case af_err::AF_ERR_DIFF_TYPE:
                name = "AF_ERR_DIFF_TYPE";
                break;
            case af_err::AF_ERR_DRIVER:
                name = "AF_ERR_DRIVER";
                break;
            case af_err::AF_ERR_INTERNAL:
                name = "AF_ERR_INTERNAL";
                break;
            case af_err::AF_ERR_INVALID_ARRAY:
                name = "AF_ERR_INVALID_ARRAY";
                break;
            case af_err::AF_ERR_LOAD_LIB:
                name = "AF_ERR_LOAD_LIB";
                break;
            case af_err::AF_ERR_LOAD_SYM:
                name = "AF_ERR_LOAD_SYM";
                break;
            case af_err::AF_ERR_NONFREE:
                name = "AF_ERR_NONFREE";
                break;
            case af_err::AF_ERR_NOT_CONFIGURED:
                name = "AF_ERR_NOT_CONFIGURED";
                break;
            case af_err::AF_ERR_NOT_SUPPORTED:
                name = "AF_ERR_NOT_SUPPORTED";
                break;
            case af_err::AF_ERR_NO_DBL:
                name = "AF_ERR_NO_DBL";
                break;
            case af_err::AF_ERR_NO_GFX:
                name = "AF_ERR_NO_GFX";
                break;
            case af_err::AF_ERR_NO_HALF:
                name = "AF_ERR_NO_HALF";
                break;
            case af_err::AF_ERR_NO_MEM:
                name = "AF_ERR_NO_MEM";
                break;
            case af_err::AF_ERR_RUNTIME:
                name = "AF_ERR_RUNTIME";
                break;
            case af_err::AF_ERR_SIZE:
                name = "AF_ERR_SIZE";
                break;
            case af_err::AF_ERR_TYPE:
                name = "AF_ERR_TYPE";
                break;
            case af_err::AF_ERR_UNKNOWN:
                name = "AF_ERR_UNKNOWN";
                break;
        }

        return format_to(ctx.out(), "(af_err) {} {}", (long long) p, name);
    }
};

#endif //__FORMATTERS_H__
