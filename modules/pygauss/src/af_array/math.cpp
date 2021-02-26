#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

typedef af_err (binary_arith_fn)(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

af::array binary_arith(const std::variant<af::array, py::float_> &left,
                       const std::variant<af::array, py::float_> &right,
                       binary_arith_fn op) {

    af_array out = nullptr;

    if (left.index() != 0 && right.index() != 0)
        throw std::runtime_error("At least one parameter must be an array");

    if (left.index() == 0 && right.index() == 0) {
        // both are an array
        check_af_error((*op)(&out,
                             std::get<af::array>(left).get(),
                             std::get<af::array>(right).get(),
                             GForStatus::get()));
    }

    auto shape = left.index() == 0 ?
                 std::get<af::array>(left).dims() :
                 std::get<af::array>(right).dims();

    auto value = left.index() != 0 ?
                 std::get<py::float_>(left) :
                 std::get<py::float_>(right);

    auto constant = constant_array(value, shape);

    if (left.index() == 0)
        check_af_error((*op)(&out, std::get<af::array>(left).get(), constant, GForStatus::get()));
    else
        check_af_error((*op)(&out, constant, std::get<af::array>(right).get(), GForStatus::get()));

    return af::array(out);
}


void math_bindings(py::module &left) {

    left.def("cbrt",
             [](const af::array &a) {
                 return af::cbrt(a);
             },
             py::arg("a").none(false),
             "Computes the cube root");

    left.def("erf",
             [](const af::array &a) {
                 return af::erf(a);
             },
             py::arg("a").none(false),
             "Computes the error function value");

    left.def("erfc",
             [](const af::array &a) {
                 return af::erfc(a);
             },
             py::arg("a").none(false),
             "Complementary Error function value");

    left.def("exp",
             [](const af::array &a) {
                 return af::exp(a);
             },
             py::arg("a").none(false),
             "Exponential of input");

    left.def("expm1",
             [](const af::array &a) {
                 return af::expm1(a);
             },
             py::arg("a").none(false),
             "Exponential of input - 1");

    left.def("factorial",
             [](const af::array &a) {
                 return af::factorial(a);
             },
             py::arg("a").none(false),
             "Factorial function");

    left.def("lgamma",
             [](const af::array &a) {
                 return af::lgamma(a);
             },
             py::arg("a").none(false),
             "Logarithm of absolute values of Gamma function");

    left.def("log",
             [](const af::array &a) {
                 return af::log(a);
             },
             py::arg("a").none(false),
             "Natural logarithm");

    left.def("log10",
             [](const af::array &a) {
                 return af::log10(a);
             },
             py::arg("a").none(false),
             "logarithm base 10");

    left.def("log1p",
             [](const af::array &a) {
                 return af::log1p(a);
             },
             py::arg("a").none(false),
             "Natural logarithm of (1 + in)");

    left.def("sqrt",
             [](const af::array &a) {
                 return af::sqrt(a);
             },
             py::arg("a").none(false),
             "Square Root");

    left.def("rsqrt",
             [](const af::array &a) {
                 return af::rsqrt(a);
             },
             py::arg("a").none(false),
             "The reciprocal or inverse square root of input arrays (1/sqrt(x))");

    left.def("tgamma",
             [](const af::array &a) {
                 return af::tgamma(a);
             },
             py::arg("a").none(false),
             "The gamma function");

    left.def("arccosh",
             [](const af::array &a) {
                 return af::acosh(a);
             },
             py::arg("a").none(false),
             "Inverse hyperbolic cosine");

    left.def("arcsinh",
             [](const af::array &a) {
                 return af::asinh(a);
             },
             py::arg("a").none(false),
             "Inverse hyperbolic sine");

    left.def("arctanh",
             [](const af::array &a) {
                 return af::atanh(a);
             },
             py::arg("a").none(false),
             "Inverse hyperbolic tangent");

    left.def("cosh",
             [](const af::array &a) {
                 return af::cosh(a);
             },
             py::arg("a").none(false),
             "Hyperbolic cosine");

    left.def("sinh",
             [](const af::array &a) {
                 return af::sinh(a);
             },
             py::arg("a").none(false),
             "Hyperbolic sine");

    left.def("tanh",
             [](const af::array &a) {
                 return af::tanh(a);
             },
             py::arg("a").none(false),
             "Hyperbolic tangent");

    left.def("abs",
             [](const af::array &a) {
                 return af::abs(a);
             },
             py::arg("a").none(false),
             "Absolute value");

    left.def("arg",
             [](const af::array &a) {
                 return af::arg(a);
             },
             py::arg("a").none(false),
             "Phase of a number in the complex plane");

    left.def("ceil",
             [](const af::array &a) {
                 return af::ceil(a);
             },
             py::arg("a").none(false),
             "Round to integer greater than equal to current value");

    left.def("floor",
             [](const af::array &a) {
                 return af::floor(a);
             },
             py::arg("a").none(false),
             "Round to integer less than equal to current value");

    left.def("round",
             [](const af::array &a) {
                 return af::round(a);
             },
             py::arg("a").none(false),
             "Round to nearest integer");

    left.def("sign",
             [](const af::array &a) {
                 return af::sign(a);
             },
             py::arg("a").none(false),
             "Checks inputs are negative");

    left.def("trunc",
             [](const af::array &a) {
                 return af::trunc(a);
             },
             py::arg("a").none(false),
             "Truncate float values.");

    left.def("arccos",
             [](const af::array &a) {
                 return af::acos(a);
             },
             py::arg("a").none(false),
             "Inverse cosine");

    left.def("arcsin",
             [](const af::array &a) {
                 return af::asin(a);
             },
             py::arg("a").none(false),
             "Inverse sine");

    left.def("arctan",
             [](const af::array &a) {
                 return af::atan(a);
             },
             py::arg("a").none(false),
             "Inverse tangent");

    left.def("cos",
             [](const af::array &a) {
                 return af::cos(a);
             },
             py::arg("a").none(false),
             "Cosine fn");

    left.def("sin",
             [](const af::array &a) {
                 return af::sin(a);
             },
             py::arg("a").none(false),
             "Sine fn");

    left.def("tan",
             [](const af::array &a) {
                 return af::tan(a);
             },
             py::arg("a").none(false),
             "Tangent fn");

    left.def("atan2",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_atan2);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Arc tan of the inputs.");

    left.def("atan2",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_atan2);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Arc tan of the inputs.");

    left.def("hypot",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_hypot);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Hypotenuse of the two inputs.");

    left.def("hypot",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_hypot);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Hypotenuse of the two inputs.");

    left.def("root",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_root);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "This function supports real inputs only. Complex inputs are not yet supported..");

    left.def("root",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_root);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "This function supports real inputs only. Complex inputs are not yet supported..");

    left.def("complex",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_cplx2);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Constructs a new complex array two independent sources.");

    left.def("complex",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_cplx2);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Constructs a new complex array two independent sources.");

    left.def("max_of",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_maxof);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Maximum of two inputs.");

    left.def("max_of",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_maxof);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Maximum of two inputs.");

    left.def("min_of",
             [](const af::array &left, const std::variant<af::array, py::float_> &right) {
                 return binary_arith(left, right, af_minof);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Minimum of two inputs.");

    left.def("min_of",
             [](const py::float_ &left, const af::array &right) {
                 return binary_arith(left, right, af_minof);
             },
             py::arg("left").none(false),
             py::arg("right").none(false),
             "Minimum of two inputs.");

    left.def("clamp",
             [](const af::array &a, const std::variant<af::array, py::float_> &lo,
                const std::variant<af::array, py::float_> &up) {
                 auto lo_a = lo.index() == 0 ?
                             std::get<af::array>(lo).get() :
                             constant_array(std::get<py::float_>(lo), a.dims());
                 auto up_a = up.index() == 0 ?
                             std::get<af::array>(up).get() :
                             constant_array(std::get<py::float_>(up), a.dims());

                 af_array out = nullptr;
                 check_af_error(af_clamp(&out, a.get(), lo_a, up_a, GForStatus::get()));
                 return af::array(out);
             },
             py::arg("a").none(false),
             py::arg("lo").none(false),
             py::arg("up").none(false),
             "Clamps the array between two values");

}