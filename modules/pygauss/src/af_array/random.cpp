#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

#define FLOAT32_EPS 1.1920929e-07

namespace py = pybind11;


#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

af::array gamma(af::randomEngine &re, double alpha, double lambda, int64_t n) {
    if (alpha < 1.0) {
        auto r = gamma(re, alpha + 1.0, lambda, n);
        return r * af::pow(af::randu(af::dim4(n), af::dtype::f32, re), 1.0 / alpha);
    }
    auto random_numbers = af::constant(0.0, n, af::dtype::f32);
    auto d = alpha - 1. / 3.;
    auto c = 1.0 / sqrt(9.0 * d);
    auto number_generated = 0L;
    auto number_generated_total = 0;
    while (number_generated < n) {
        auto number_left = n - number_generated;
        auto z = af::randn(af::dim4(number_left), af::dtype::f32, re);
        auto y = (1.0 + c * z);
        auto v = y * y * y;
        auto accept_index_1 = ((z >= -1.0 / c) && (v > 0.0));
        auto z_accept_1 = z(accept_index_1);
        auto v_accept_1 = v(accept_index_1).copy();
        auto u_accept_1 = af::randu(af::dim4(v_accept_1.elements()), af::dtype::f32, re);
        auto accept_index_2 =
                u_accept_1 < af::exp((0.5 * z_accept_1 * z_accept_1 + d - d * v_accept_1 + d * af::log(v_accept_1)));
        auto x_accept = d * v_accept_1(accept_index_2) / lambda;
        auto number_accept = x_accept.elements();

        auto left_index = af::seq(number_generated, std::min(n, number_generated + number_accept));
        auto right_index = af::seq(0, std::min(number_left, number_accept));
        random_numbers(left_index) = x_accept(right_index);
        number_generated += number_accept;
        number_generated_total += number_left;
    }
    return random_numbers;
}

#pragma clang diagnostic pop


af::array wald(af::randomEngine &re, double mu, double lambda, int64_t n) {
    auto y = af::pow(af::randn(af::dim4(n), af::dtype::f32, re), 2);
    auto u = af::randu(af::dim4(n), af::dtype::f32, re);
    auto mu2 = std::pow(mu, 2);
    auto x = mu + mu2 / (2.0 * lambda) * y - mu / (2.0 * lambda) * sqrt(4.0 * mu * lambda * y + mu2 * y * y);
    auto reject_index = u > (mu / (mu + x));
    x(reject_index) = mu2 / x(reject_index);
    return x;
}


af::array logistic(af::randomEngine &re, double loc, double scale, int64_t n) {
    auto u = af::randu(af::dim4(n), af::dtype::f32, re);
    u = af::min(u, 1.0 - FLOAT32_EPS);
    return loc - (scale * af::log(1.0 / u - 1.0));
}

void random_bindings(py::module &m) {

    py::class_<af::randomEngine>(m, "ShapeletsRandomEngine")
            .def("uniform",
                 [](af::randomEngine &self,
                    const double low, const double high,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     af::array v = af::randu(shape, dtype, self);
                     return (v * (high - low)) + low;
                 },
                 py::arg("low") = 0.0,
                 py::arg("high") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Samples are uniformly distributed over the half-open interval [low, high)")
            .def("exponential",
                 [](af::randomEngine &self,
                    const double scale, const af::dim4 &shape, const af::dtype &dtype) {
                     auto v = af::randu(shape, dtype, self);
                     v = af::min(v, 1.0 - FLOAT32_EPS);
                     return af::log(1.0 - v) * (-scale);
                 },
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from an exponential distribution.  Scale is the inverse of lambda.")
            .def("gamma",
                 [](af::randomEngine &self, const double alpha, const double scale,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     auto l = 1.0 / scale;
                     auto r = gamma(self, alpha, l, shape.elements());
                     if (r.type() != dtype)
                         r = r.as(dtype);
                     return af::moddims(r, shape);
                 },
                 py::arg("alpha").none(false),
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Gamma distribution.  Alpha is what is called "
                 "shape or K parameter of the Gamma distribution")
            .def("chisquare",
                 [](af::randomEngine &self, double df,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     auto a = df / 2.0;
                     auto l = 1.0 / 2.0;
                     auto r = gamma(self, a, l, shape.elements());
                     if (r.type() != dtype)
                         r = r.as(dtype);
                     return af::moddims(r, shape);
                 },
                 py::arg("df").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a chi-square distribution.  df is the degree of freedom")
            .def("beta",
                 [](af::randomEngine &self, const double a, const double b,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     auto x = gamma(self, a, 1.0, shape.elements());
                     auto y = gamma(self, b, 1.0, shape.elements());
                     auto r = x / (x + y);
                     if (r.type() != dtype)
                         r = r.as(dtype);
                     return af::moddims(r, shape);
                 },
                 py::arg("a").none(false),
                 py::arg("b").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Beta distribution.  Alpha is what is called "
                 "shape or K parameter of the Gamma distribution")
            .def("wald",
                 [](af::randomEngine &self, const double mean, const double scale,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     auto r = wald(self, mean, scale, shape.elements());
                     if (r.type() != dtype)
                         r = r.as(dtype);
                     return af::moddims(r, shape);
                 },
                 py::arg("mean").none(false),
                 py::arg("scale").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Wald, or inverse Gaussian, distribution.")
            .def("normal",
                 [](af::randomEngine &self, const double loc, const double scale,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     return loc + (scale * af::randn(shape, dtype, self));
                 },
                 py::arg("loc") = 0.0,
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw random samples from a normal (Gaussian) distribution.")
            .def("standard_normal",
                 [](af::randomEngine &self,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     return af::randn(shape, dtype, self);
                 },
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw random samples from a normal (Gaussian) distribution.")
            .def("lognormal",
                 [](af::randomEngine &self, const double mean, const double sigma,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     return af::exp(mean + (sigma * af::randn(shape, dtype, self)));
                 },
                 py::arg("mean") = 0.0,
                 py::arg("sigma") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a log-normal distribution.")
            .def("logistic",
                 [](af::randomEngine &self, const double loc, const double scale,
                    const af::dim4 &shape, const af::dtype &dtype) {
                     auto r = logistic(self, loc, scale, shape.elements());
                     if (r.type() != dtype)
                         r = r.as(dtype);
                     return af::moddims(r, shape);
                 },
                 py::arg("loc") = 0.0,
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a logistic distribution.")
            .def("multivariate_normal",
                 [](af::randomEngine &self,
                    const py::object &mean,
                    const py::object &cov,
                    const int64_t samples,
                    const af::dtype &dtype) {

                     auto m = array_like::to_array(mean);
                     auto cv = array_like::to_array(cov);

                     if (!m.has_value())
                         std::runtime_error("Unable to process mean as N vector");
                     if (!cv.has_value())
                         std::runtime_error("Unable to process mean as NxN vector");

                     if (m->elements() != m->dims(0))
                         std::runtime_error("mean parameter should be a vector");

                     if (cv->dims(0) != cv->dims(1) && m->elements() != cv->dims(0))
                         std::runtime_error("cov matrix dimension doesn't match mean vector dimensions");

                     auto z = af::randn(af::dim4(samples, m->elements()), m->type(), self);
                     af::array cv_cholesky;
                     af::cholesky(cv_cholesky, cv.value());
                     return af::matmulNT(z, cv_cholesky) + m.value();
                 },
                 py::arg("mean").none(false),
                 py::arg("cov").none(false),
                 py::arg("samples").none(false),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a logistic distribution.");


    py::enum_<af::randomEngineType>(m, "RandomEngineType", "Built-in engines for random number generation")
            .value("Default", af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT, "Defaults to Philox")
            .value("Mersenne", af::randomEngineType::AF_RANDOM_ENGINE_MERSENNE, "Mersenne GP 11213.")
            .value("Threefry", af::randomEngineType::AF_RANDOM_ENGINE_THREEFRY, "Threefry 2X32_16.")
            .value("Philox", af::randomEngineType::AF_RANDOM_ENGINE_PHILOX, "Philox 4x32_10.")
            .export_values();

    m.def("default_rng",
          [](const af::randomEngineType type, const unsigned long long seed) {
              return af::randomEngine(type, seed);
          },
          py::arg("type") = af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT,
          py::arg("seed") = 0ULL,
          "Creates a new random engine");

    m.def("randu",
          [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              if (engine.has_value())
                  return af::randu(shape, dtype, engine.value());
              return af::randu(shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          py::arg("engine") = py::none(),
          "Creates a new array using random values drawn from a uniform distribution");

    m.def("randn", [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              if (engine.has_value())
                  return af::randn(shape, dtype, engine.value());
              return af::randn(shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          py::arg("engine") = py::none(),
          "Creates a new array using random numbers drawn from a normal distribution");

    m.def("randint",
          [](const int64_t low, const std::optional<int64_t> high,
             const af::dim4 &shape, const af::dtype &dtype,
             std::optional<af::randomEngine> &engine) {

              auto l = 0.;
              auto h = static_cast<double>(low);
              if (high.has_value()) {
                  l = static_cast<double>(low);
                  h = static_cast<double>(high.value());
              }
              auto divisor = 1.0 / (h - l);
              auto u = engine.has_value() ?
                       af::randu(shape, af::dtype::f32, engine.value()) :
                       af::randu(shape, af::dtype::f32);

              u = af::min(u, 1.0 - FLOAT32_EPS) / divisor;
              return u.as(dtype);
          },
          py::arg("low").none(false),
          py::arg("high") = py::none(),
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::s32,
          py::arg("engine") = py::none(),
          "Return random integers from low (inclusive) to high (exclusive).\n"
          "\n"
          "Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” "
          "interval [low, high). If high is None (the default), then results are from [0, low).");
}
