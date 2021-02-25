#ifndef AF_ARRAY_PYBIND11_CONVERSIONS__H
#define AF_ARRAY_PYBIND11_CONVERSIONS__H

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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