#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gauss.h>
#include <pygauss.h>

#define FLOAT32_EPS 1.1920929e-07

namespace py = pybind11;

void pygauss::bindings::random_numbers(py::module &m) {

    py::enum_<af::randomEngineType>(m, "RandomEngineType", "Built-in engines for random number generation")
            .value("Default", af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT, "Defaults to Philox")
            .value("Mersenne", af::randomEngineType::AF_RANDOM_ENGINE_MERSENNE, "Mersenne GP 11213.")
            .value("Threefry", af::randomEngineType::AF_RANDOM_ENGINE_THREEFRY, "Threefry 2X32_16.")
            .value("Philox", af::randomEngineType::AF_RANDOM_ENGINE_PHILOX, "Philox 4x32_10.")
            .export_values();

    py::class_<af::randomEngine>(m, "ShapeletsRandomEngine")
            .def("uniform",
                 [](af::randomEngine &self, const double low, const double high, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::uniform(low, high, shape, dtype, self);
                 },
                 py::arg("low") = 0.0,
                 py::arg("high") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Samples are uniformly distributed over the half-open interval [low, high)")

            .def("exponential",
                 [](af::randomEngine &self, const double scale, const af::dim4 &shape, const af::dtype &dtype) {
                     return gauss::random::exponential(scale, shape, dtype, self);
                 },
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from an exponential distribution.  Scale is the inverse of lambda.")

            .def("gamma",
                 [](af::randomEngine &self, const double alpha, const double scale, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::gamma(alpha, 1.0 / scale, shape, dtype, self);
                 },
                 py::arg("alpha").none(false),
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Gamma distribution.  Alpha is what is called "
                 "shape or K parameter of the Gamma distribution")

            .def("chisquare",
                 [](af::randomEngine &self, double df, const af::dim4 &shape, const af::dtype &dtype) {
                     return gauss::random::chisquare(df, shape, dtype, self);
                 },
                 py::arg("df").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a chi-square distribution.  df is the degree of freedom")

            .def("beta",
                 [](af::randomEngine &self, const double a, const double b, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::beta(a, b, shape, dtype, self);
                 },
                 py::arg("a").none(false),
                 py::arg("b").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Beta distribution.  Alpha is what is called "
                 "shape or K parameter of the Gamma distribution")

            .def("wald",
                 [](af::randomEngine &self, const double mean, const double scale, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::wald(mean, scale, shape, dtype, self);
                 },
                 py::arg("mean").none(false),
                 py::arg("scale").none(false),
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a Wald, or inverse Gaussian, distribution.")

            .def("normal",
                 [](af::randomEngine &self, const double loc, const double scale, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::normal(loc, scale, shape, dtype, self);
                 },
                 py::arg("loc") = 0.0,
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw random samples from a normal (Gaussian) distribution.")

            .def("standard_normal",
                 [](af::randomEngine &self, const af::dim4 &shape, const af::dtype &dtype) {
                     return gauss::random::normal(0.0, 1.0, shape, dtype, self);
                 },
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw random samples from a normal (Gaussian) distribution.")

            .def("lognormal",
                 [](af::randomEngine &self, const double mean, const double sigma, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::lognormal(mean, sigma, shape, dtype, self);
                 },
                 py::arg("mean") = 0.0,
                 py::arg("sigma") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a log-normal distribution.")

            .def("logistic",
                 [](af::randomEngine &self, const double loc, const double scale, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::logistic(loc, scale, shape, dtype, self);
                 },
                 py::arg("loc") = 0.0,
                 py::arg("scale") = 1.0,
                 py::arg("shape") = af::dim4(1, 1, 1, 1),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a logistic distribution.")

            .def("multivariate_normal",
                 [](af::randomEngine &self, const py::object &mean, const py::object &cov, const int64_t samples,
                    const af::dtype &dtype) {
                     auto m = pygauss::arraylike::cast(mean).as(dtype);
                     auto cv = pygauss::arraylike::cast(cov).as(dtype);
                     return gauss::random::multivariate_normal(samples, m, cv, self);
                 },
                 py::arg("mean").none(false),
                 py::arg("cov").none(false),
                 py::arg("samples").none(false),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a logistic distribution.");


    m.def("default_rng",
          [](const af::randomEngineType type, const unsigned long long seed) {
              return af::randomEngine(type, seed);
          },
          py::arg("type") = af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT,
          py::arg("seed") = 0ULL,
          "Creates a new random engine");


    m.def("random",
          [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              return gauss::random::uniform(0.0, 1.0, shape, dtype, engine);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          py::arg("engine") = py::none(),
          "Creates a new array using random values drawn from a uniform distribution");

    m.def("randn", [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              return gauss::random::normal(0.0, 1.0, shape, dtype, engine);
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

              u = l + (af::min(u, 1.0 - FLOAT32_EPS) / divisor);
              return u.as(dtype);
          },
          py::arg("low").none(false),
          py::arg("high") = py::none(),
          py::arg("shape") = af::dim4(1, 1),
          py::arg("dtype") = af::dtype::s32,
          py::arg("engine") = py::none(),
          "Return random integers from low (inclusive) to high (exclusive).\n"
          "\n"
          "Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” "
          "interval [low, high). If high is None (the default), then results are from [0, low).");
}
