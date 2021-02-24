#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void construction_bindings(py::module &m) {

    m.def("array",
          &af_from_array_like,
          py::arg("array_like").none(false),
          py::arg("shape").none(true) = py::none(),
          py::arg("dtype").none(true) = py::none());

    m.def("transpose",
          [](const py::object &arr_like, const std::optional<af::dim4> &shape, const std::optional<af::dtype> &dtype) {
              auto arr = af_from_array_like(arr_like, shape, dtype);
              return af::transpose(arr, false);
          },py::arg("array_like").none(false),
          py::arg("shape").none(true) = py::none(),
          py::arg("dtype").none(true) = py::none()
    );

    m.def("empty",
          [](const af::dim4 &shape, const af::dtype &type) {
              return af::array(shape, type);
          },
          py::arg("shape").none(false),
          py::arg("type") = af::dtype::f32
    );

    m.def("constant",
          [](const af::dim4 &shape, const py::object &value, const std::optional<af::dtype> &type) {
              return af::array(constant_array(value, shape, type));
          },
          py::arg("shape").none(false),
          py::arg("value").none(false),
          py::arg("type").none(true) = af::dtype::f32,
          "Create a array from a scalar input value");

    m.def("identity",
          [](const af::dim4 &shape, const af::dtype &type) {
              return af::identity(shape, type);
          },
          py::arg("shape").none(false),
          py::arg("type") = af::dtype::f32,
          "Creates an identity array with diagonal values set to one.");

    m.def("iota",
          [](const af::dim4 &shape, const af::dim4 &tile, const af::dtype &type) {
              return af::iota(shape, tile, type);
          },
          py::arg("shape").none(false),
          py::arg("tile") = 1,
          py::arg("type") = af::dtype::f32,
          "Create an sequence [0, shape.elements() - 1] and modify to specified dimensions dims "
          "and then tile it according to tile");

    m.def("range", [](const af::dim4 &shape, int seq_dim, const af::dtype &type) {
              return af::range(shape, seq_dim, type);
          },
          py::arg("shape").none(false),
          py::arg("seq_dim") = -1,
          py::arg("type") = af::dtype::f32,
          "Creates an array with [0, n] values along the seq_dim which is tiled across other dimensions.");
}