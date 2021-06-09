/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

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

    UNARY_TEMPLATE_FN_LAMBDA(isfinite, is_finite)
    UNARY_TEMPLATE_FN(isinf, af_isinf)
    UNARY_TEMPLATE_FN(isnan, af_isnan)
    // isnat
    // isneginf
    // isposinf



    UNARY_TEMPLATE_FN_LAMBDA(iscomplex, is_complex)
    UNARY_TEMPLATE_FN_LAMBDA(isreal, is_real)
    // These need direct support in the python layer
    // iscomplexobj
    // isfortran
    // isreal
    // isrealobj
    // isscalar


    BINARY_TEMPLATE_FN(logical_and, af_and, false)
    BINARY_TEMPLATE_FN(logical_or, af_or, false)
    UNARY_TEMPLATE_FN(logical_not, af_not)
    // MISSING LOGICAL_XOR


    BINARY_TEMPLATE_FN(equal, af_eq, false)
    BINARY_TEMPLATE_FN(not_equal, af_neq, false)
    BINARY_TEMPLATE_FN(greater, af_gt, false)
    BINARY_TEMPLATE_FN(greater_equal, af_ge,false)
    BINARY_TEMPLATE_FN(less, af_lt, false)
    BINARY_TEMPLATE_FN(less_equal, af_le,false)

    // allclose
    // isclose
    // array_equiv
    // array_equal

}
