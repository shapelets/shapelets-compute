#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <pygauss.h>

namespace spd = spdlog;
namespace py = pybind11;

af::array rad2deg(const af::array &in) {
    return in * (180.0 / 3.14159265358979323846);
}

af::array deg2rad(const af::array &in) {
    return in * (3.14159265358979323846 / 180.0);
}

af::array signbit(const af::array &in) {
    return in < 0;
}

af::array reciprocal(const af::array &in) {
    return 1.0 / in;
}

af::array positive(const af::array &in) {
    return in.copy();
}

af::array negative(const af::array &in) {
    return -in;
}

af::array angle_deg(const af::array &in) {
    return rad2deg(af::arg(in));
}


af::array floor_divide(const af::array &left, const af::array &right, bool broadcast) {
    auto previous = af::gforGet();
    if (!previous && broadcast)
        af::gforSet(true);
    auto result = af::floor(left / right);
    if (!previous && broadcast)
        af::gforSet(false);
    return result;
}


void pygauss::bindings::math_operations(py::module &m) {

//
// Trigonometric Functions
//
    UNARY_TEMPLATE_FN(sin, af_sin)
    UNARY_TEMPLATE_FN(cos, af_cos)
    UNARY_TEMPLATE_FN(tan, af_tan)
    UNARY_TEMPLATE_FN(arcsin, af_asin)
    UNARY_TEMPLATE_FN(arccos, af_acos)
    UNARY_TEMPLATE_FN(arctan, af_atan)
    BINARY_TEMPLATE_FN(hypot, af_hypot, true)
    BINARY_TEMPLATE_FN(arctan2, af_atan2, true)
    UNARY_TEMPLATE_FN_LAMBDA(degrees, rad2deg)
    UNARY_TEMPLATE_FN_LAMBDA(rad2deg, rad2deg)
    UNARY_TEMPLATE_FN_LAMBDA(radians, deg2rad)
    UNARY_TEMPLATE_FN_LAMBDA(deg2rad, deg2rad)
    // missing unwrap https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap

//
// Hyperbolic functions
    UNARY_TEMPLATE_FN(sinh, af_sinh)
    UNARY_TEMPLATE_FN(cosh, af_cosh)
    UNARY_TEMPLATE_FN(tanh, af_tanh)
    UNARY_TEMPLATE_FN(arcsinh, af_asinh)
    UNARY_TEMPLATE_FN(arccosh, af_acosh)
    UNARY_TEMPLATE_FN(arctanh, af_atanh)

//
// Rounding
//
    UNARY_TEMPLATE_FN(trunc, af_trunc)
    UNARY_TEMPLATE_FN(floor, af_floor)
    UNARY_TEMPLATE_FN(ceil, af_ceil)
    UNARY_TEMPLATE_FN(rint, af_round)
    UNARY_TEMPLATE_FN(fix, af_trunc)

    m.def("round",
          [](const py::object &array_like, const int decimals) {

              std::optional<af::array> result = std::nullopt;
              auto arr = arraylike::as_array_checked(array_like);

              if (decimals == 0)
                  return af::round(arr);

              auto scale = pow(10, decimals);
              return af::round(arr * scale) / scale;
          },
          py::arg("array_like").none(false),
          py::arg("decimals") = 0);

//
// Sums, products, differences
//
//  prod        -> see algorithms
//  sum         -> see algorithms
//  nanprod     -> see algorithms
//  nansum      -> see algorithms
//  cumprod     -> see algorithms
//  cumsum      -> see algorithms
//  nancumprod  -> see algorithms
//  nancumsum   -> see algorithms
//  diff        -> see algorithms
//  ediff1d     -> see algorithms
//
//  gradient    -> missing
//  cross       -> missing
//  trapz       -> missing

//
// Exponents and logarithms
//
// logaddexp    --> missing
// logaddexp2   --> missing


    UNARY_TEMPLATE_FN(exp, af_exp)
    UNARY_TEMPLATE_FN(expm1, af_expm1)
    UNARY_TEMPLATE_FN(exp2, af_pow2)
    UNARY_TEMPLATE_FN(log, af_log)
    UNARY_TEMPLATE_FN(log10, af_log10)
    UNARY_TEMPLATE_FN(log2, af_log10)
    UNARY_TEMPLATE_FN(log1p, af_log1p)


//
// Other special functions
//
// i0x  --> missing
// sinc --> missing

//
// Floating point routines
//
// copysign -> missing
// frexp -> missing
// ldexp -> missing
// nextafter -> missing
// spacing -> missing

    UNARY_TEMPLATE_FN_LAMBDA(signbit, signbit)

//
// Rational routines
//
// lcm -> missing
// gcd -> missing


//
// Arithmetic operations
//
// float_power  --> missing
// fmod         --> missing
// modf         --> missing
// divmod       --> missing

    BINARY_TEMPLATE_FN(add, af_add, false)
    UNARY_TEMPLATE_FN_LAMBDA(reciprocal, reciprocal)
    UNARY_TEMPLATE_FN_LAMBDA(positive, positive)
    UNARY_TEMPLATE_FN_LAMBDA(negative, negative)
    BINARY_TEMPLATE_FN(multiply, af_mul, false)
    BINARY_TEMPLATE_FN(divide, af_div, false)
    BINARY_TEMPLATE_FN(true_divide, af_div, false)
    BINARY_TEMPLATE_FN(power, af_pow, false)
    BINARY_TEMPLATE_FN(substract, af_sub, false)
    BINARY_TEMPLATE_FN_LAMBDA(floor_divide, floor_divide, false)
    BINARY_TEMPLATE_FN(mod, af_mod, false)
    BINARY_TEMPLATE_FN(rem, af_rem, false)

//
// Handling complex numbers
//
//  angle --> missing

    UNARY_TEMPLATE_FN(real, af_real)
    UNARY_TEMPLATE_FN(imag, af_imag)
    UNARY_TEMPLATE_FN(conjugate, af_conjg)
    UNARY_TEMPLATE_FN(complex, af_cplx)
    UNARY_TEMPLATE_FN(angle, af_arg)
    UNARY_TEMPLATE_FN_LAMBDA(angle_deg, angle_deg)
    // this one is not in np
    BINARY_TEMPLATE_FN(complex, af_cplx2, false)


//
// Miscellaneous
//
//  convolve --> see signal processing
// heavyside
// real_if_close

    UNARY_TEMPLATE_FN(sqrt, af_sqrt)
    UNARY_TEMPLATE_FN(cbrt, af_cbrt)
    UNARY_TEMPLATE_FN(square, af_pow2)
    UNARY_TEMPLATE_FN(absolute, af_abs)
    UNARY_TEMPLATE_FN(fabs, af_abs)
    UNARY_TEMPLATE_FN(sign, af_sign)

    m.def("clip",
          [](const py::object &array_like, const py::object &lo, const py::object &up) {
              auto a = pygauss::arraylike::as_array_checked(array_like);
              if (up.is_none() && lo.is_none())
                  return a;

              if (!up.is_none() && !lo.is_none()) {
                  auto u = pygauss::arraylike::is_scalar(up) ?
                           pygauss::arraylike::scalar_as_array_checked(up, a.dims(), a.type()) :
                           pygauss::arraylike::as_array_checked(up);

                  auto l = pygauss::arraylike::is_scalar(lo) ?
                           pygauss::arraylike::scalar_as_array_checked(lo, a.dims(), a.type()) :
                           pygauss::arraylike::as_array_checked(lo);

                  af_array out = nullptr;
                  throw_on_error(af_clamp(&out, a.get(), l.get(), u.get(), GForStatus::get()));
                  return af::array(out);
              }

              if (!up.is_none()) {
                  auto u = pygauss::arraylike::is_scalar(up) ?
                           pygauss::arraylike::scalar_as_array_checked(up, a.dims(), a.type()) :
                           pygauss::arraylike::as_array_checked(up);

                  af_array out = nullptr;
                  throw_on_error(af_minof(&out, a.get(), u.get(), GForStatus::get()));
                  return af::array(out);
              }

              auto l = pygauss::arraylike::is_scalar(lo) ?
                       pygauss::arraylike::scalar_as_array_checked(lo, a.dims(), a.type()) :
                       pygauss::arraylike::as_array_checked(lo);

              af_array out = nullptr;
              throw_on_error(af_maxof(&out, a.get(), l.get(), GForStatus::get()));
              return af::array(out);
          },
          py::arg("array_like").none(false),
          py::arg("lo") = py::none(),
          py::arg("up") = py::none());


//
// Additional Fns
//

    UNARY_TEMPLATE_FN(sigmoid, af_sigmoid)
    UNARY_TEMPLATE_FN(erf, af_erf)
    UNARY_TEMPLATE_FN(erfc, af_erfc)
    UNARY_TEMPLATE_FN(rsqrt, af_rsqrt)
    UNARY_TEMPLATE_FN(factorial, af_factorial)
    UNARY_TEMPLATE_FN(tgamma, af_tgamma)
    UNARY_TEMPLATE_FN(lgamma, af_lgamma)
    BINARY_TEMPLATE_FN(root, af_root, false)


//
// Elementwise bit operations
//


    BINARY_TEMPLATE_FN(bitwise_and, af_bitand, false)
    BINARY_TEMPLATE_FN(bitwise_or, af_bitor, false)
    BINARY_TEMPLATE_FN(bitwise_xor, af_bitxor, false)
    BINARY_TEMPLATE_FN(left_shift, af_bitshiftl, false)
    BINARY_TEMPLATE_FN(right_shift, af_bitshiftr, false)
//  Missing in my mac build
//  UNARY_TEMPLATE_FN(invert, af_bitnot)
}
