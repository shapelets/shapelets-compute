#include <arrayfire.h>
#include <pybind11/pybind11.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void complex_bindings(py::module &m) {

    // complex from two arrays construction in module math
    // as it takes profit of the c implementation with
    // the parallel flag.

//    m.def("conjg",
//          [](const af::array &a) {
//              return af::conjg(a);
//          },
//          py::arg("a").none(false),
//          "Gets the complex conjugate");
//
//    m.def("imag",
//          [](const af::array &a) {
//              return af::imag(a);
//          },
//          py::arg("a").none(false),
//          "Extracts the imaginary part of a complex array or matrix");
//
//    m.def("real",
//          [](const af::array &a) {
//              return af::real(a);
//          },
//          py::arg("a").none(false),
//          "Extracts the real part of a complex array or matrix");
}
