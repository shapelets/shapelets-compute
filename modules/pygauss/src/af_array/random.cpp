#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void random_bindings(py::module &m) {

    py::class_<af::randomEngine>(m, "ShapeletsRandomEngine"); // NOLINT(bugprone-unused-raii)

    py::enum_<af::randomEngineType>(m, "RandomEngineType", "Built-in engines for random number generation")
            .value("Default", af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT, "Defaults to Philox")
            .value("Mersenne", af::randomEngineType::AF_RANDOM_ENGINE_MERSENNE, "Mersenne GP 11213.")
            .value("Threefry", af::randomEngineType::AF_RANDOM_ENGINE_THREEFRY, "Threefry 2X32_16.")
            .value("Philox", af::randomEngineType::AF_RANDOM_ENGINE_PHILOX, "Philox 4x32_10.")
            .export_values();

    m.def("random_engine",
          [](const af::randomEngineType type, const unsigned long long seed) {
              return af::randomEngine(type, seed);
          },
          py::arg("type") = af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT,
          py::arg("seed") = 0,
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
}