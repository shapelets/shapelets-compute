/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
            case af::dtype::c64:
                return 16;
            case af::dtype::u64:
            case af::dtype::s64:
            case af::dtype::f64:
            case af::dtype::c32:
            default:
                return 8;
        }
    }

    bool is_scalar(const py::object &value) {
        return detail::python_is_scalar(value)
               || detail::numpy_is_scalar(value);
    }

    /**
     * Ensures an array is of a floating type, coverting if necessary.
     * @param src
     * @return
     */
    void ensure_floating(af::array& src, bool warn_if_conversion) {
        if (src.isfloating())
            return;

        auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
        if (warn_if_conversion)
            PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 2);

        src = src.as(safe_conversion);
    }

    /**
     * Tries to convert obj to a valid array
     *
     * @return  If the conversion is successful, the optional result will contain a
     *          valid array.  Otherwise, std::nullopt
     */
    std::optional<af::array> as_array(const py::object& obj) {
        if (detail::af_is_array(obj))
            return detail::from_af_array(obj);

        if (detail::numpy_is_array(obj))
            return detail::numpy_array_to_array(obj);

        return std::nullopt;
    }

    af::array as_itself_or_promote(const py::object& obj, const af::dim4 &shape, const af::dtype& ref_type) {
        std::optional<af::array> candidate = std::nullopt;
        if (detail::af_is_array(obj))
            candidate = detail::from_af_array(obj);
        else if (detail::numpy_is_array(obj))
            candidate = detail::numpy_array_to_array(obj);

        if (candidate)
            return candidate.value();

        if (detail::python_is_scalar(obj))
            candidate = detail::python_scalar_to_array(obj, shape, ref_type);
        else if (detail::numpy_is_scalar(obj))
            candidate = detail::numpy_scalar_to_array(obj, shape, ref_type);

        if (!candidate)
            throw std::invalid_argument("Unable to process object as valid scalar or tensor");

        return candidate.value();
    }

    af::array as_array_like(const py::object& obj, const std::optional<af::dim4> &shape,
                            const std::optional<af::dtype> &dtype) {

        std::optional<af::array> candidate = std::nullopt;
        if (detail::af_is_array(obj))
            candidate = detail::from_af_array(obj);
        else if (detail::numpy_is_array(obj))
            candidate = detail::numpy_array_to_array(obj);

        if (candidate.has_value()) {
            auto result = candidate.value();
            if (shape && (result.dims() != shape.value())) {
                if (result.elements() == shape->elements())
                    result = af::moddims(result, shape.value());
                else
                    throw std::invalid_argument("Unable to meet shape requirements as element count doesn't match");
            }

            if (dtype && (result.type() != dtype.value()))
                result = result.as(dtype.value());

            return result;
        }

        if (!shape)
            throw std::invalid_argument("Shape is required to promote a scalar to tensor");

        if (detail::python_is_scalar(obj))
            candidate = detail::python_scalar_to_array(obj, shape.value(), dtype.value_or(af::dtype::f32));
        else if (detail::numpy_is_scalar(obj))
            candidate = detail::numpy_scalar_to_array(obj, shape.value(), dtype.value_or(af::dtype::f32));

        if (!candidate)
            throw std::invalid_argument("Unable to process scalar value");

        auto result = candidate.value();
        if (dtype && (result.type() != dtype.value()))
            result = result.as(dtype.value());

        return result;
    }

    af::array as_array_like(const py::object& obj, const af::array& like) {
        std::optional<af::array> candidate = std::nullopt;

        if (detail::af_is_array(obj))
            candidate = detail::from_af_array(obj);
        else if (detail::numpy_is_array(obj))
            candidate = detail::numpy_array_to_array(obj);
        else if (detail::python_is_scalar(obj))
            candidate = detail::python_scalar_to_array(obj, like.dims(), like.type());
        else if (detail::numpy_is_scalar(obj))
            candidate = detail::numpy_scalar_to_array(obj, like.dims(), like.type());

        if (!candidate)
            throw std::invalid_argument("Unable to convert to array");

        auto result = candidate.value();

        if (result.dims() != like.dims()) {
            if (result.elements() != like.elements())
                throw std::invalid_argument("Unable to change shape as element count differs");

            result = af::moddims(result, like.dims());
        }

        if (result.type() != like.type()) {
            result = result.as(like.type());
        }

        return result;
    }

    af::array as_array_checked(const py::object& obj) {
        auto conv = as_array(obj);
        if (!conv)
            throw std::invalid_argument("Unable to convert to array");
        return conv.value();
    }

    /**
     * Builds an array from a scalar value to the dimensions specified in shape
     */
    std::optional<af::array> scalar_as_array(const py::object& obj, const af::dim4& shape, const af::dtype& ref_type) {
        if (detail::python_is_scalar(obj))
            return detail::python_scalar_to_array(obj, shape, ref_type);

        if (detail::numpy_is_scalar(obj))
            return detail::numpy_scalar_to_array(obj, shape, ref_type);

        return std::nullopt;
    }

    af::array scalar_as_array_checked(const py::object& obj, const af::dim4& shape, const af::dtype& ref_type) {
        auto conv = scalar_as_array(obj, shape, ref_type);
        if (!conv)
            throw std::invalid_argument("Unable to convert to array");
        return conv.value();
    }


    /**
     * Transforms both entries in such way that:
     *  - If both elements are arrays, they will be transformed independently and returned
     *    without modifications.
     *  - If one of the elements are arrays, but the other is a constant, then, the array
     *    will be converted without changes and the constant will be converted to an array
     *    with the same dimensionality as the other one.
     *  - If both are constants, this method will return std::nullopt
     *
     *  No attempt is made to change the types of each individual arrays
     *
     * @return   The pair respects the order of the arguments (x will be the first element of
     *           the resulting pair.
     */
    std::optional<std::pair<af::array, af::array>>
    as_array(const py::object &x, const py::object &y) {
        auto x_as_array = as_array(x);
        auto y_as_array = as_array(y);

        // Both are not an array, return with std::nullopt
        if (!x_as_array && !y_as_array)
            return std::nullopt;

        // Both are arrays, nothing else to do
        if (x_as_array && y_as_array)
            return std::make_pair(x_as_array.value(), y_as_array.value());

        if (x_as_array)
            y_as_array = scalar_as_array(y, x_as_array->dims(), x_as_array->type());
        else
            x_as_array = scalar_as_array(x, y_as_array->dims(), y_as_array->type());

        // if either is not an array, fail with std::nullopt
        if (!x_as_array || !y_as_array)
            return std::nullopt;

        // finally, return the two arrays
        return std::make_pair(x_as_array.value(), y_as_array.value());
    }


//
//        if (dtype.has_value()) {
//            if (candidate.type() != dtype.value()) {
//                candidate = candidate.as(dtype.value());
//            }
//        }
//
//        if (shape.has_value()) {
//            if (candidate.dims() != shape.value()) {
//                if (candidate.elements() == shape.value().elements()) {
//                    candidate = af::moddims(candidate, shape.value());
//                }
//                else {
//                    std::ostringstream stm;
//                    stm << "Unable to adjust shape of tensor since the number of elements are different. ";
//                    stm << "Source has " << candidate.elements() << " and target requires " << shape.value().elements();
//                    auto err_msg = stm.str();
//                    spd::error(err_msg);
//                    throw std::runtime_error(err_msg);
//                }
//            }
//        }
//
//
//
//        std::ostringstream stm;
//        stm << "Unable to process " << py::repr(array_like) << " as a valid tensor.";
//        auto err_msg = stm.str();
//        spd::error(err_msg);
//        throw std::runtime_error(err_msg);


}
