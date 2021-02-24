#include <arrayfire.h>
#include <pybind11/pybind11.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void informative_bindings(py::module &m) {

    m.def("isNaN",
          [](const af::array &a) {
              return af::isNaN(a);
          },
          py::arg("a").none(false),
          "Returns a new array with all its positions set to 0, except on those where the value is NaN");

    m.def("isInf",
          [](const af::array &a) {
              return af::isInf(a);
          },
          py::arg("a").none(false),
          "Returns a new array with all its positions set to 0, except on those where the value is Inf");

    m.def("isZero",
          [](const af::array &a) {
              return af::iszero(a);
          },
          py::arg("a").none(false),
          "Returns a new array with all its positions set to 0, except on those where the value is zero");
}