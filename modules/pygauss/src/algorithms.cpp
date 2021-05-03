#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pygauss.h>

namespace py = pybind11;

typedef af_err (*reduce_all_op)(double *, double *, af_array);

typedef af_err (*reduce_all_nan_op)(double *, double *, af_array, double);

typedef af_err (*reduce_dim_op)(af_array *, af_array, const int);

typedef af_err (*reduce_dim_nan_op)(af_array *, af_array, const int, const double);

std::complex<double> reduce_all_complex(const af::array &a, const reduce_all_op op)
{
    double real, imag;
    pygauss::throw_on_error((*op)(&real, &imag, a.get()));
    return std::complex<double>(real, imag);
}

std::complex<double> reduce_all_complex_nan(const af::array &a, const double nan, const reduce_all_nan_op op)
{
    double real, imag;
    pygauss::throw_on_error((*op)(&real, &imag, a.get(), nan));
    return std::complex<double>(real, imag);
}

double reduce_all_real(const af::array &a, const reduce_all_op op)
{
    double real, imag;
    pygauss::throw_on_error((*op)(&real, &imag, a.get()));
    return real;
}

double reduce_all_real_nan(const af::array &a, const double nan, const reduce_all_nan_op op)
{
    double real, imag;
    pygauss::throw_on_error((*op)(&real, &imag, a.get(), nan));
    return real;
}

af::array reduce_dim(const af::array &a, const int dim, const reduce_dim_op op)
{
    af_array result = nullptr;
    pygauss::throw_on_error((*op)(&result, a.get(), dim));
    return af::array(result);
}

af::array reduce_dim_nan(const af::array &a, const int dim, double nan, const reduce_dim_nan_op op)
{
    af_array result = nullptr;
    pygauss::throw_on_error((*op)(&result, a.get(), dim, nan));
    return af::array(result);
}

af::array minof_no_nan(af::array &a, af::array &b, bool broadcast)
{
    // choose a sensible max value to replace the nans
    double max = std::numeric_limits<float>::max();
    if (a.isdouble())
        max = std::numeric_limits<double>::max();
    else if (a.ishalf())
        max = 65504.0f;

    // Values of "a" are replaced with corresponding values of max, when cond is false.
    af::replace(a, !af::isNaN(a), max);
    af::replace(b, !af::isNaN(b), max);
    af_array out = nullptr;
    pygauss::throw_on_error(af_minof(&out, a.get(), b.get(), broadcast));
    return af::array(out);
}

af::array maxof_no_nan(af::array &a, af::array &b, bool broadcast)
{
    // choose a sensible max value to replace the nans
    double min = -std::numeric_limits<float>::max();
    if (a.isdouble())
        min = -std::numeric_limits<double>::max();
    else if (a.ishalf())
        min = -65504.0f;

    // Values of "a" are replaced with corresponding values of max, when cond is false.
    af::replace(a, !af::isNaN(a), min);
    af::replace(b, !af::isNaN(b), min);
    af_array out = nullptr;
    pygauss::throw_on_error(af_maxof(&out, a.get(), b.get(), broadcast));
    return af::array(out);
}

typedef std::variant<int, float, std::complex<float>, std::complex<double>, py::array> MyType;

void pygauss::bindings::parallel_algorithms(py::module &m)
{

    //
    // all and any from logic
    //
    m.def(
        "any",
        [](const py::object &array_like, const std::optional<int> &dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            std::variant<bool, af::array> result;
            if (!dim.has_value())
                result = reduce_all_real(a, af_any_true_all) == 1.0;
            else
                result = reduce_dim(a, dim.value(), af_any_true);

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "Check if any the elements along a specified dimension are true.");

    m.def(
        "all",
        [](const py::object &array_like, const std::optional<int> &dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            std::variant<bool, af::array> result;
            if (!dim.has_value())
                result = reduce_all_real(a, af_all_true_all) == 1.0;
            else
                result = reduce_dim(a, dim.value(), af_all_true);

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "Check if all the elements along a specified dimension are true.");

    m.def(
        "nan_to_num",
        [](const py::object &array_like, double nan, double inf) {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // Values of "a" are replaced with corresponding values of nan, when cond is false.
            af::replace(a, !nanLocations, nan);
            // array containing 1's where input is Inf, and 0 otherwise.
            auto infLocations = af::isInf(a);
            // Values of "a" are replaced with corresponding values of inf, when cond is false.
            af::replace(a, !infLocations, inf);
            // return cleaned version of array_like
            return a;
        },
        py::arg("array_like").none(false),
        py::arg("nan") = 0.0,
        py::arg("inf") = 0.0,
        "Return the minimum of an array or minimum along an axis, propagating NaNs");

    //
    // amin, nanmin, minimum, fmin, argmin
    //
    // amin -> Return the minimum of an array or minimum along an axis, propagating NaNs
    // nanmin -> The minimum value of an array along a given axis, ignoring any NaNs.
    // minimum -> Element-wise minimum of two arrays, propagating any NaNs.
    // fmin -> Element-wise minimum of two arrays, ignoring any NaNs.
    // argmin -> Return the indices of the minimum values.
    //

    typedef std::variant<std::complex<double>, af::array, py::float_> numberOrArray;
    typedef std::variant<std::tuple<unsigned int, std::complex<double>>, std::tuple<af::array, af::array>, std::tuple<unsigned int, py::float_>> indexAndValues;

    m.def(
        "amin",
        [](const py::object &array_like, const std::optional<int> &dim) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            numberOrArray result;

            if (!dim.has_value()) // NOLINT(bugprone-branch-clone)
                if (a.iscomplex())
                    result = reduce_all_complex(a, af_min_all);
                else
                    result = py::float_(reduce_all_real(a, af_min_all));
            else
                result = reduce_dim(a, dim.value(), af_min);

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "Return the minimum of an array or minimum along an axis, propagating NaNs");

    m.def(
        "nanmin",
        [](const py::object &array_like, const std::optional<int> &dim) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // choose a sensible max value to replace the nans
            double max = std::numeric_limits<float>::max();
            if (a.isdouble())
                max = std::numeric_limits<double>::max();
            else if (a.ishalf())
                max = 65504.0f;

            // Values of "a" are replaced with corresponding values of max, when cond is false.
            af::replace(a, !nanLocations, max);

            numberOrArray result;
            if (!dim.has_value()) // NOLINT(bugprone-branch-clone)
                if (a.iscomplex())
                    result = reduce_all_complex(a, af_min_all);
                else
                    result = py::float_(reduce_all_real(a, af_min_all));
            else
                result = reduce_dim(a, dim.value(), af_min);
            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "The minimum value of an array along a given axis, ignoring any NaNs.");

    BINARY_TEMPLATE_FN(minimum, af_minof, "Minimum of two inputs, with NaNs propagated.")

    BINARY_TEMPLATE_FN_LAMBDA(fmin, minof_no_nan, "Minimum of two inputs, ignoring NaNs")

    m.def(
        "argmin",
        [](const py::object &array_like, const std::optional<int> &dim) -> indexAndValues {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            if (!dim.has_value())
            {
                double real, imag;
                unsigned int index;
                throw_on_error(af_imin_all(&real, &imag, &index, a.get()));

                indexAndValues result;
                if (a.iscomplex())
                    result = std::make_tuple(index, std::complex<double>(real, imag));
                else
                    result = std::make_tuple(index, py::float_(real));
                return result;
            }

            af_array out = nullptr;
            af_array index = nullptr;
            throw_on_error(af_imin(&out, &index, a.get(), dim.value()));
            return std::make_tuple(af::array(index), af::array(out));
        },
        py::arg("array_like").none(false),
        py::arg("dim") = py::none(),
        "Returns the indices and values of the minimum values along an axis, propagating NaNs");

    m.def(
        "nanargmin",
        [](const py::object &array_like, const std::optional<int> &dim) -> indexAndValues {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            // choose a sensible max value to replace the nans
            double min = -std::numeric_limits<float>::max();
            if (a.isdouble())
                min = -std::numeric_limits<double>::max();
            else if (a.ishalf())
                min = -65504.0f;

            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);

            // Values of "a" are replaced with corresponding values of max, when cond is false.
            af::replace(a, !nanLocations, min);

            if (!dim.has_value())
            {
                double real, imag;
                unsigned int index;
                throw_on_error(af_imin_all(&real, &imag, &index, a.get()));
                indexAndValues result;
                if (a.iscomplex())
                    result = std::make_tuple(index, std::complex<double>(real, imag));
                else
                    result = std::make_tuple(index, py::float_(real));
                return result;
            }

            af_array out = nullptr;
            af_array index = nullptr;
            throw_on_error(af_imin(&out, &index, a.get(), dim.value()));
            return std::make_tuple(af::array(index), af::array(out));
        },
        py::arg("array_like").none(false),
        py::arg("dim") = py::none(),
        "Returns the indices and values of the minimum values along an axis, ignoring NaN");

    //
    // amax, nanmax, maximum, fmax, argmax
    //
    // Same as before, but with max
    //

    m.def(
        "amax",
        [](const py::object &array_like, const std::optional<int> &dim) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            numberOrArray result;
            if (!dim.has_value()) // NOLINT(bugprone-branch-clone)
                if (a.iscomplex())
                    result = reduce_all_complex(a, af_max_all);
                else
                    result = py::float_(reduce_all_real(a, af_max_all));
            else
                result = reduce_dim(a, dim.value(), af_max);

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "Return the maximum of an array or minimum along an axis, propagating NaNs");

    m.def(
        "nanmax",
        [](const py::object &array_like, const std::optional<int> &dim) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // choose a sensible max value to replace the nans
            double min = -std::numeric_limits<float>::max();
            if (a.isdouble())
                min = -std::numeric_limits<double>::max();
            else if (a.ishalf())
                min = -65504.0f;

            // Values of "a" are replaced with corresponding values of max, when cond is false.
            af::replace(a, !nanLocations, min);

            numberOrArray result; 
            if (!dim.has_value()) // NOLINT(bugprone-branch-clone)
                if (a.iscomplex())
                    result = reduce_all_complex(a, af_max_all);
                else
                    result = py::float_(reduce_all_real(a, af_max_all));
            else 
                result = reduce_dim(a, dim.value(), af_max);

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "The maximum value of an array along a given axis, ignoring any NaNs.");

    BINARY_TEMPLATE_FN(maximum, af_maxof, "Maximum of two inputs, with NaNs propagated.")

    BINARY_TEMPLATE_FN_LAMBDA(fmax, maxof_no_nan, "Maximum of two inputs, ignoring NaNs")

    m.def(
        "argmax",
        [](const py::object &array_like, const std::optional<int> &dim) -> indexAndValues {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            if (!dim.has_value())
            {
                indexAndValues result;
                double real, imag;
                unsigned int index;
                throw_on_error(af_imax_all(&real, &imag, &index, a.get()));
                if (a.iscomplex()) 
                    result = std::make_tuple(index, std::complex<double>(real, imag));
                else
                    result = std::make_tuple(index, py::float_(real));
                return result;
            }

            af_array out = nullptr;
            af_array index = nullptr;
            throw_on_error(af_imax(&out, &index, a.get(), dim.value()));
            return std::make_tuple(af::array(index), af::array(out));
        },
        py::arg("array_like").none(false),
        py::arg("dim") = py::none(),
        "Returns the indices and values of the maximum values along an axis, with NaNs propagated.");

    m.def(
        "nanargmax",
        [](const py::object &array_like, const std::optional<int> &dim) -> indexAndValues {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            // choose a sensible max value to replace the nans
            double min = -std::numeric_limits<float>::max();
            if (a.isdouble())
                min = -std::numeric_limits<double>::max();
            else if (a.ishalf())
                min = -65504.0f;

            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);

            // Values of "a" are replaced with corresponding values of max, when cond is false.
            af::replace(a, !nanLocations, min);

            if (!dim.has_value())
            {
                double real, imag;
                unsigned int index;
                throw_on_error(af_imax_all(&real, &imag, &index, a.get()));
                indexAndValues result;
                if (a.iscomplex()) 
                    result = std::make_tuple(index, std::complex<double>(real, imag));
                else
                    result = std::make_tuple(index, py::float_(real));
                return result;
            }

            af_array out = nullptr;
            af_array index = nullptr;
            throw_on_error(af_imax(&out, &index, a.get(), dim.value()));
            return std::make_tuple(af::array(index), af::array(out));
        },
        py::arg("array_like").none(false),
        py::arg("dim") = py::none(),
        "Returns the indices and values of the maximum values along an axis, with NaNs propagated.");

    m.def(
        "count_nonzero",
        [](const py::object &array_like, const std::optional<int> &dim) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            numberOrArray result;
            if (!dim.has_value())
                result = py::float_(reduce_all_real(a, af_count_all));
            else
                result = reduce_dim(a, dim.value(), af_count);
            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        "Count the number of non zero elements in an array along a specified dimension");

    m.def(
        "sum",
        [](const py::object &array_like, const std::optional<int> &dim,
           const std::optional<double> &nan_value) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            numberOrArray result;

            if (!dim.has_value())
            {
                if (!nan_value.has_value())
                    if (a.iscomplex())
                        result = reduce_all_complex(a, af_sum_all);
                    else
                        result = py::float_(reduce_all_real(a, af_sum_all));
                else if (a.iscomplex())
                    result = reduce_all_complex_nan(a, nan_value.value(), af_sum_nan_all);
                else
                    result = py::float_(reduce_all_real_nan(a, nan_value.value(), af_sum_nan_all));
            }
            else
            {
                if (!nan_value.has_value())
                    result = reduce_dim(a, dim.value(), af_sum);
                else
                    result = reduce_dim_nan(a, dim.value(), nan_value.value(), af_sum_nan);
            }

            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        py::arg("nan_value").none(true) = py::none(),
        "Calculate the sum of all the elements along a specified dimension.\n"
        "\n"
        "This function is equivalent to both sum and nansum in numpy; simply set the value "
        "of `nan_value` parameter to either include NaNs in the sumation (None) or replace "
        "NaN values with your choice.");

    m.def(
        "product",
        [](const py::object &array_like, const std::optional<int> &dim,
           const std::optional<double> &nan_value) -> numberOrArray {
            auto a = pygauss::arraylike::as_array_checked(array_like);

            numberOrArray result;
            if (!dim.has_value())
            {
                if (!nan_value.has_value())
                    if (a.iscomplex())
                        result = reduce_all_complex(a, af_product_all);
                    else
                        result = py::float_(reduce_all_real(a, af_product_all));
                else if (a.iscomplex())
                    result = reduce_all_complex_nan(a, nan_value.value(), af_product_nan_all);
                else
                    result = py::float_(reduce_all_real_nan(a, nan_value.value(), af_product_nan_all));
            }
            else
            {
                if (!nan_value.has_value())
                    result = reduce_dim(a, dim.value(), af_product);
                else
                    result = reduce_dim_nan(a, dim.value(), nan_value.value(), af_product_nan);
            }
            return result;
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(true) = py::none(),
        py::arg("nan_value").none(true) = py::none(),
        "Calculate the product of all the elements along a specified dimension.\n"
        "\n"
        "This function is equivalent to both prod and nanprod in numpy; simply set the value "
        "of `nan_value` parameter to either include NaNs in the multiplication (None) or replace "
        "NaN values with your choice.");

    m.def(
        "cumsum",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return reduce_dim(a, dim, af_accum);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        "Cumulative sum of an array along a specified dimension, propagating NaNs");

    m.def(
        "nancumsum",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // Values of "a" are replaced with zero, when cond is false.
            af::replace(a, !nanLocations, 0.0);
            return reduce_dim(a, dim, af_accum);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        "Cumulative sum of an array along a specified dimension, ignoring NaNs");

    m.def(
        "cumprod",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return af::scan(a, dim, AF_BINARY_MUL, true);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        "Cumulative product of an array along a specified dimension, propagating NaNs");

    m.def(
        "nancumprod",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // Values of "a" are replaced with zero, when cond is false.
            af::replace(a, !nanLocations, 0.0);

            return af::scan(a, dim, AF_BINARY_MUL, true);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        "Cumulative product of an array along a specified dimension, ignoring NaNs");

    m.def(
        "scan",
        [](const py::object &array_like, int dim, const af::binaryOp &op, const bool &inclusive_scan) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return af::scan(a, dim, op, inclusive_scan);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        py::arg("op") = af::binaryOp::AF_BINARY_ADD,
        py::arg("inclusive_scan") = true,
        "Generalized scan of an array, which can the operations defined in ScanOp.");
    //
    m.def(
        "nanscan",
        [](const py::object &array_like, int dim, double nan, const af::binaryOp &op, const bool &inclusive_scan) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            // array containing 1's where input is NaN, and 0 otherwise.
            auto nanLocations = af::isNaN(a);
            // Values of "a" are replaced with zero, when cond is false.
            af::replace(a, !nanLocations, nan);
            return af::scan(a, dim, op, inclusive_scan);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        py::arg("nan") = 0.0,
        py::arg("op") = af::binaryOp::AF_BINARY_ADD,
        py::arg("inclusive_scan") = true,
        "Generalized scan of an array, which can the operations defined in ScanOp; this function replaces NaN "
        "values with the value provided in parameter `nan`, which defaults to 0.0");

    m.def(
        "diff1",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return reduce_dim(a, dim, af_diff1);
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(false),
        "Find the first order differences along specified dimensions");

    m.def(
        "diff2",
        [](const py::object &array_like, int dim) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return reduce_dim(a, dim, af_diff2);
        },
        py::arg("array_like").none(false),
        py::arg("dim").none(false),
        "Find the second order differences along specified dimensions");

    m.def(
        "sort",
        [](const py::object &array_like, int dim, const bool asc) -> af::array {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return af::sort(a, dim, asc);
        },
        py::arg("array_like").none(false),
        py::arg("dim") = 0,
        py::arg("asc") = true,
        "Sort the array along a specified dimension");

    m.def(
        "sort_index",
        [](const py::object &data, int dim, const bool asc) -> std::tuple<af::array, af::array> {
            auto a = pygauss::arraylike::as_array_checked(data);
            af::array sorted_keys, sorted_data;
            af::sort(sorted_keys, sorted_data, a, dim, asc);

            return {sorted_data, sorted_keys};
        },
        py::arg("data").none(false),
        py::arg("dim") = 0,
        py::arg("asc") = true);

    m.def(
        "sort_by_key",
        [](const py::object &keys, const py::object &data, int dim, const bool asc) -> std::tuple<af::array, af::array> {
            auto a = pygauss::arraylike::as_array_checked(data);
            auto k = pygauss::arraylike::as_array_checked(keys);

            af::array sorted_keys, sorted_data;
            af::sort(sorted_keys, sorted_data, k, a, dim, asc);

            return {sorted_data, sorted_keys};
        },
        py::arg("keys").none(false),
        py::arg("data").none(false),
        py::arg("dim") = 0,
        py::arg("asc") = true);

    m.def(
        "flatnonzero",
        [](const py::object &array_like) -> af::array {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return af::where(a);
        },
        py::arg("array_like").none(false),
        "Return indices that are non-zero in the flattened version of a");

    m.def(
        "unique",
        [](const py::object &array_like, const bool &is_sorted) {
            auto a = pygauss::arraylike::as_array_checked(array_like);
            return af::setUnique(a, is_sorted);
        },
        py::arg("array_like").none(false),
        py::arg("is_sorted") = py::bool_(false),
        "Find the unique elements of an array.");

    m.def(
        "union",
        [](const py::object &x1, const py::object &x2, const bool &is_unique) {
            auto a = pygauss::arraylike::as_array_checked(x1);
            auto b = pygauss::arraylike::as_array_checked(x2);
            return af::setUnion(a, b, is_unique);
        },
        py::arg("x1").none(false),
        py::arg("x2").none(false),
        py::arg("is_unique") = py::bool_(false),
        "Find the union of two arrays.");

    m.def(
        "intersect",
        [](const py::object &x1, const py::object &x2, const bool &is_unique) {
            auto a = pygauss::arraylike::as_array_checked(x1);
            auto b = pygauss::arraylike::as_array_checked(x2);
            return af::setIntersect(a, b, is_unique);
        },
        py::arg("x1").none(false),
        py::arg("x2").none(false),
        py::arg("is_unique") = py::bool_(false),
        "Find the union of two arrays.");

    m.def(
        "any_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            if (k.type() != af::dtype::s32 || k.type() != af::dtype::u32)
                k = k.as(af::dtype::s32);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::anyTrueByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "all_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);

            if (k.type() != af::dtype::s32 || k.type() != af::dtype::u32)
                k = k.as(af::dtype::s32);

            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::allTrueByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "count_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::countByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "max_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::maxByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "min_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::minByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "product_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::productByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "nanproduct_by_key",
        [](const py::object &keys, const py::object &vals, const double nan_value, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::productByKey(out_keys, out_vals, k, v, dim.value_or(-1), nan_value);
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("nan_value") = 1.0,
        py::arg("dim") = py::none());

    m.def(
        "sum_by_key",
        [](const py::object &keys, const py::object &vals, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::sumByKey(out_keys, out_vals, k, v, dim.value_or(-1));
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("dim") = py::none());

    m.def(
        "nansum_by_key",
        [](const py::object &keys, const py::object &vals, const double nan_value, const std::optional<int> &dim) -> std::tuple<af::array, af::array> {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            af::array out_keys, out_vals;
            af::sumByKey(out_keys, out_vals, k, v, dim.value_or(-1), nan_value);
            return {out_keys, out_vals};
        },
        py::arg("keys").none(false),
        py::arg("vals").none(true),
        py::arg("nan_value") = 0.0,
        py::arg("dim") = py::none());

    m.def(
        "scan_by_key",
        [](const py::object &keys, const py::object &vals, int dim, const af::binaryOp &op, const bool &inclusive_scan) {
            auto k = pygauss::arraylike::as_array_checked(keys);
            auto v = pygauss::arraylike::as_array_checked(vals);
            return af::scanByKey(k, v, dim, op, inclusive_scan);
        },
        py::arg("keys").none(false),
        py::arg("vals").none(false),
        py::arg("dim") = 0,
        py::arg("op") = af::binaryOp::AF_BINARY_ADD,
        py::arg("inclusive_scan") = true);
}
