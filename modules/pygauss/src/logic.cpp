#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <pygauss.h>

namespace spd = spdlog;
namespace py = pybind11;

af::array is_complex(const af::array &in) {
    if (!in.iscomplex())
        return af::constant(false, in.dims(), af::dtype::b8);

    return af::imag(in) != 0;
}

af::array is_real(const af::array &in) {
    if (!in.iscomplex())
        return af::constant(true, in.dims(), af::dtype::b8);

    return af::imag(in) == 0;
}

af::array is_finite(const af::array &in) {
    return af::iszero(af::isNaN(in) || af::isInf(in)).as(af::dtype::b8);
}

void pygauss::bindings::logic_operations(py::module &m) {

    //
    // all -> algorithms
    // any -> algorithms

    UNARY_TEMPLATE_FN_LAMBDA(isfinite, is_finite, "Returns a boolean tensor where all zero positions are set to True")
    UNARY_TEMPLATE_FN(isinf, af_isinf, "Returns a boolean tensor where all infinite positions are set to True")
    UNARY_TEMPLATE_FN(isnan, af_isnan, "Returns a boolean tensor where all nan positions are set to True")
    // isnat
    // isneginf
    // isposinf



    UNARY_TEMPLATE_FN_LAMBDA(iscomplex, is_complex, "Returns a bool array, where True if input element is complex.")
    UNARY_TEMPLATE_FN_LAMBDA(isreal, is_real, "Returns a bool array, where True if input element is complex.")
    // These need direct support in the python layer
    // iscomplexobj
    // isfortran
    // isreal
    // isrealobj
    // isscalar


    BINARY_TEMPLATE_FN(logical_and, af_and, "Performs element-wise logical and.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(logical_or, af_or, "Performs element-wise logical or.  The result is always a boolean tensor")
    UNARY_TEMPLATE_FN(logical_not, af_not,
                      "Performs element-wise logical complement.  The result is always a boolean tensor")
    // MISSING LOGICAL_XOR


    BINARY_TEMPLATE_FN(equal, af_eq, "Element-wise equality test.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(not_equal, af_neq, "Element-wise inequality test.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(greater, af_gt, "Element-wise greater than test.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(greater_equal, af_ge,
                       "Element-wise greater than or equal test.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(less, af_lt, "Element-wise less than test.  The result is always a boolean tensor")
    BINARY_TEMPLATE_FN(less_equal, af_le,
                       "Element-wise less than or equal test.  The result is always a boolean tensor")

    // allclose
    // isclose
    // array_equiv
    // array_equal

}
