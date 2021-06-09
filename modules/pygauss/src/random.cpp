/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

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
                 [](af::randomEngine &self, const double alpha, const double scale,
                    const af::dim4 &shape, const af::dtype &dtype) {
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
                 [](af::randomEngine &self, const double mean, const double sigma, const af::dim4 &shape,
                    const af::dtype &dtype) {
                     return gauss::random::normal(mean, sigma, shape, dtype, self);
                 },
                 py::arg("mean") = 0.0,
                 py::arg("sigma") = 1.0,
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
                 [](af::randomEngine &self, const py::object &mean, const py::object &cov,
                    const int64_t samples, const af::dtype &dtype) {

                     auto m = arraylike::as_array_checked(mean).as(dtype);
                     auto cv = arraylike::as_array_checked(cov).as(dtype);

                     return gauss::random::multivariate_normal(samples, m, cv, self);
                 },
                 py::arg("mean").none(false),
                 py::arg("cov").none(false),
                 py::arg("samples").none(false),
                 py::arg("dtype") = af::dtype::f32,
                 "Draw samples from a logistic distribution.")
            .def("permutation",
                 [](af::randomEngine &self, const py::object &x, const int32_t axis) -> af::array {
                     auto is_scalar = arraylike::is_scalar(x);
                     auto original = is_scalar ?
                                     af::iota(af::dim4(py::cast<dim_t>(x)), af::dim4(1), af::dtype::s32) :
                                     arraylike::as_array_checked(x);

                     auto checked_axis = is_scalar ? 0 : axis;
                     return gauss::random::permute(original, checked_axis, self);
                 },
                 py::arg("x").none(false),
                 py::arg("axis") = 0);

    m.def("default_rng",
          [](const af::randomEngineType type, const unsigned long long seed) {
              return af::randomEngine(type, seed);
          },
          py::arg("type") = af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT,
          py::arg("seed") = 0ULL);

    m.def("permutation",
          [](const py::object &x, const int32_t axis, std::optional<af::randomEngine> &engine) -> af::array {
              auto is_scalar = arraylike::is_scalar(x);
              auto original = is_scalar ?
                              af::iota(af::dim4(py::cast<dim_t>(x)), af::dim4(1), af::dtype::s32) :
                              arraylike::as_array_checked(x);

              auto checked_axis = is_scalar ? 0 : axis;
              return gauss::random::permute(original, checked_axis, engine);
          },
          py::arg("x").none(false),
          py::arg("axis") = 0,
          py::arg("engine") = py::none());

    m.def("random",
          [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              return gauss::random::uniform(0.0, 1.0, shape, dtype, engine);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          py::arg("engine") = py::none());

    m.def("randn", [](const af::dim4 &shape, const af::dtype &dtype, std::optional<af::randomEngine> &engine) {
              return gauss::random::normal(0.0, 1.0, shape, dtype, engine);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          py::arg("engine") = py::none());

    m.def("randint",
          [](const int64_t low, const std::optional<int64_t> high, const af::dim4 &shape, const af::dtype &dtype,
             std::optional<af::randomEngine> &engine) {
              return gauss::random::randint(low, high, shape, dtype, engine);
          },
          py::arg("low").none(false),
          py::arg("high") = py::none(),
          py::arg("shape") = af::dim4(1, 1),
          py::arg("dtype") = af::dtype::s32,
          py::arg("engine") = py::none());
}
