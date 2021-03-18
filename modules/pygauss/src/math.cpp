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
    UNARY_TEMPLATE_FN(sin, af_sin, "Trigonometric sine element-wise")
    UNARY_TEMPLATE_FN(cos, af_cos, "Trigonometric cosine element-wise")
    UNARY_TEMPLATE_FN(tan, af_tan, "Trigonometric tangent element-wise")
    UNARY_TEMPLATE_FN(arcsin, af_asin, "Trigonometric inverse sine element-wise")
    UNARY_TEMPLATE_FN(arccos, af_acos, "Trigonometric inverse cosine element-wise")
    UNARY_TEMPLATE_FN(arctan, af_atan, "Trigonometric inverse tangent element-wise")
    BINARY_TEMPLATE_FN(hypot, af_hypot, "Given the sides of a triangle, returns the hypotenuse")
    BINARY_TEMPLATE_FN(arctan2, af_atan2, "Element-wise arc tangent of the inputs")
    UNARY_TEMPLATE_FN_LAMBDA(degrees, rad2deg, "Radians to degrees, element-wise")
    UNARY_TEMPLATE_FN_LAMBDA(rad2deg, rad2deg, "Radians to degrees, element-wise")
    UNARY_TEMPLATE_FN_LAMBDA(radians, deg2rad, "Degrees to radians, element-wise")
    UNARY_TEMPLATE_FN_LAMBDA(deg2rad, deg2rad, "Degrees to radians, element-wise")
    // missing unwrap https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap

//
// Hyperbolic functions
    UNARY_TEMPLATE_FN(sinh, af_sinh, "Hyperbolic sine, element-wise.")
    UNARY_TEMPLATE_FN(cosh, af_cosh, "Hyperbolic cosine, element-wise.")
    UNARY_TEMPLATE_FN(tanh, af_tanh, "Hyperbolic tangent element-wise.")
    UNARY_TEMPLATE_FN(arcsinh, af_asinh, "Inverse hyperbolic sine, element-wise.")
    UNARY_TEMPLATE_FN(arccosh, af_acosh, "Inverse hyperbolic cosine, element-wise.")
    UNARY_TEMPLATE_FN(arctanh, af_atanh, "Inverse hyperbolic tangent, element-wise.")

//
// Rounding
//
    UNARY_TEMPLATE_FN(trunc, af_trunc, "Return the truncated value of the input, element-wise.")
    UNARY_TEMPLATE_FN(floor, af_floor, "Return the floor of the input, element-wise.")
    UNARY_TEMPLATE_FN(ceil, af_ceil, "Return the ceiling of the input, element-wise.")
    UNARY_TEMPLATE_FN(rint, af_round, "Round elements of the array to the nearest integer.")
    UNARY_TEMPLATE_FN(fix, af_trunc, "Round to nearest integer towards zero.")

    m.def("round",
          [](const py::object &array_like, const int decimals) {

              std::optional<af::array> result = std::nullopt;
              auto arr = arraylike::as_array_checked(array_like);

              if (decimals == 0)
                  return af::round(arr);

              auto scale = pow(10, decimals);
              return af::round((arr * scale) / scale);
          },
          py::arg("array_like").none(false),
          py::arg("decimals") = 0,
          "Evenly round to the given number of decimals.");

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


    UNARY_TEMPLATE_FN(exp, af_exp, "Calculate the exponential of all elements in the input array")
    UNARY_TEMPLATE_FN(expm1, af_expm1, "Calculate exp(x) - 1 for all elements in the array")
    UNARY_TEMPLATE_FN(exp2, af_pow2, "Calculate 2**p for all p in the input array.")
    UNARY_TEMPLATE_FN(log, af_log, "Natural logarithm")
    UNARY_TEMPLATE_FN(log10, af_log10, "Logarithm base 10")
    UNARY_TEMPLATE_FN(log2, af_log10, "Logarithm base 2")
    UNARY_TEMPLATE_FN(log1p, af_log1p, "Natural logarithm of (1 + in)")


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

    UNARY_TEMPLATE_FN_LAMBDA(signbit, signbit, "Returns element-wise True where signbit is set (less than zero)")

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

    BINARY_TEMPLATE_FN(add, af_add, "Add arguments element-wise ")
    UNARY_TEMPLATE_FN_LAMBDA(reciprocal, reciprocal, "Returns the reciprocal of the argument (1/x), element wise")
    UNARY_TEMPLATE_FN_LAMBDA(positive, positive, "Numerical positive, element-wise.")
    UNARY_TEMPLATE_FN_LAMBDA(negative, negative, "Numerical negative, element-wise.")
    BINARY_TEMPLATE_FN(multiply, af_mul, "Multiply arguments element-wise.")
    BINARY_TEMPLATE_FN(divide, af_div, "Returns a true division of the inputs, element-wise.")
    BINARY_TEMPLATE_FN(true_divide, af_div, "Returns a true division of the inputs, element-wise.")
    BINARY_TEMPLATE_FN(power, af_pow, "First array elements raised to powers from second array, element-wise.")
    BINARY_TEMPLATE_FN(substract, af_sub, "Subtract arguments, element-wise.")
    BINARY_TEMPLATE_FN_LAMBDA(floor_divide, floor_divide,
                              "Return the largest integer smaller or equal to the division of the inputs.")
    BINARY_TEMPLATE_FN(mod, af_mod, "Return element-wise remainder of division.")
    BINARY_TEMPLATE_FN(rem, af_rem, "Return element-wise remainder of division.")

//
// Handling complex numbers
//
//  angle --> missing

    UNARY_TEMPLATE_FN(real, af_real, "Extracts the real part of a complex array or matrix")
    UNARY_TEMPLATE_FN(imag, af_imag, "Extracts the imaginary part of a complex array or matrix")
    UNARY_TEMPLATE_FN(conj, af_conjg, "Gets the complex conjugate")
    UNARY_TEMPLATE_FN(conjugate, af_conjg, "Gets the complex conjugate")
    UNARY_TEMPLATE_FN(complex, af_cplx, "Builds a complex tensor from a real one.")
    UNARY_TEMPLATE_FN(angle, af_arg, "Returns the angle in radians")
    UNARY_TEMPLATE_FN_LAMBDA(angle_deg, angle_deg, "Returns the angle in degrees")
    // this one is not in np
    BINARY_TEMPLATE_FN(complex, af_cplx2, "Constructs a new complex array two independent sources")


//
// Miscellaneous
//
//  convolve --> see signal processing
// heavyside
// real_if_close

    UNARY_TEMPLATE_FN(sqrt, af_sqrt, "Return the non-negative square-root of an array, element-wise.")
    UNARY_TEMPLATE_FN(cbrt, af_cbrt, "Return the cube-root of an array, element-wise.")
    UNARY_TEMPLATE_FN(square, af_pow2, "Return the element-wise square of the input.")
    UNARY_TEMPLATE_FN(absolute, af_abs, "Calculate the absolute value element-wise.")
    UNARY_TEMPLATE_FN(fabs, af_abs, "Calculate the absolute value element-wise.")
    UNARY_TEMPLATE_FN(sign, af_sign, "Returns an element-wise indication of the sign of a number.")

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
                  throw_on_error(af_maxof(&out, a.get(), u.get(), GForStatus::get()));
                  return af::array(out);
              }

              auto l = pygauss::arraylike::is_scalar(lo) ?
                       pygauss::arraylike::scalar_as_array_checked(lo, a.dims(), a.type()) :
                       pygauss::arraylike::as_array_checked(lo);

              af_array out = nullptr;
              throw_on_error(af_minof(&out, a.get(), l.get(), GForStatus::get()));
              return af::array(out);
          },
          py::arg("array_like").none(false),
          py::arg("lo") = py::none(),
          py::arg("up") = py::none(),
          "Clip (limit) the values in an array.");


//
// Additional Fns
//

    UNARY_TEMPLATE_FN(sigmoid, af_sigmoid, "Sigmoid function")
    UNARY_TEMPLATE_FN(erf, af_erf, "Computes the error function value")
    UNARY_TEMPLATE_FN(erfc, af_erfc, "Complementary Error function value")
    UNARY_TEMPLATE_FN(rsqrt, af_rsqrt, "The reciprocal or inverse square root of input arrays (1/sqrt(x))")
    UNARY_TEMPLATE_FN(factorial, af_factorial, "Factorial function")
    UNARY_TEMPLATE_FN(tgamma, af_tgamma, "Gamma function")
    UNARY_TEMPLATE_FN(lgamma, af_lgamma, "Logarithm of absolute values of Gamma function")
    BINARY_TEMPLATE_FN(root, af_root, "Root function")


//
// Elementwise bit operations
//


    BINARY_TEMPLATE_FN(bitwise_and, af_bitand, "Element-wise bitwise and.")
    BINARY_TEMPLATE_FN(bitwise_or, af_bitor, "Element-wise bitwise or")
    BINARY_TEMPLATE_FN(bitwise_xor, af_bitxor, "Element-wise bitwise xor")
    BINARY_TEMPLATE_FN(left_shift, af_bitshiftl, "Element-wise shift to the left")
    BINARY_TEMPLATE_FN(right_shift, af_bitshiftr, "Element-wise shift to the right")
//  Missing in my mac build
//  UNARY_TEMPLATE_FN(invert, af_bitnot, "Compute bit-wise inversion, or bit-wise NOT, element-wise.")
}
