#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

namespace py = pybind11;


void pygauss::bindings::extract_and_transform_operations(py::module &m) {



    m.def("pad",
          [](const af::array &a, const af::dim4 &begin, const af::dim4 &end, const af::borderType fill_type) {
              return af::pad(a, begin, end, fill_type);
          },
          py::arg("a").none(false),
          py::arg("begin").none(false),
          py::arg("end").none(false),
          py::arg("fill_type").none(false),
          R"__( Pads an array

          Ensure the tuples for begin and end are specific as they represent the increase at the beginning and
          the ending; if you want to add one row at the beginning and one row to the end use (1,0,0,0) and
          (1,0,0,0) as parameters.

          )__");

    m.def("lower",
          [](const af::array &a, bool unit_diag) {
              return af::lower(a, unit_diag);
          },
          py::arg("a").none(false),
          py::arg("unit_diag") = false,
          "Create a lower triangular matrix from input array\n"
          "The parameter unit_diag forces the diagonal elements to be one.");

    m.def("upper", [](const af::array &a, bool unit_diag) {
              return af::upper(a, unit_diag);
          },
          py::arg("a").none(false),
          py::arg("unit_diag") = false,
          "Create a upper triangular matrix from input array\n"
          "The parameter unit_diag forces the diagonal elements to be one.");

    m.def("moddims",
          [](const af::array &a, const af::dim4 &shape) {
              return af::moddims(a, shape);
          },
          py::arg("a").none(false),
          py::arg("shape").none(false),
          "Changes the dimensions of an array without changing the data");

    m.def("flat", [](const af::array &a) {
              return af::flat(a);
          },
          py::arg("a").none(false),
          "It flattens an array to one dimension");

    m.def("flip",
          [](const af::array &a, const uint dimension) {
              return af::flip(a, dimension);
          },
          py::arg("a").none(false),
          py::arg("dimension") = 0,
          "Flips an array along a dimension");

    m.def("join",
          [](const std::vector<af::array *> &lst, const int dimension) {
              std::vector<af_array> handles;
              std::transform(lst.begin(), lst.end(), std::back_inserter(handles), [](const af::array *a) {
                  return a->get();
              });

              af_array out = nullptr;
              auto err = af_join_many(&out, dimension, handles.size(), handles.data());

              if (err != AF_SUCCESS)
                  throw std::runtime_error("Unable to perform join");
              return af::array(out);
          },
          py::arg("lst").none(false),
          py::arg("dimension") = 0,
          "Joins up to 10 arrays along a particular dimension");

    m.def("reorder",
          [](const af::array &a, const uint x, const uint y, const uint z, const uint w) {
              return af::reorder(a, x, y, z);
          },
          py::arg("a").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 2,
          py::arg("w") = 3,
          "It modifies the order of data within an array by exchanging data according to the change "
          "in dimensionality. The linear ordering of data within the array is preserved.");

    m.def("where",
          [](af::array &a, const af::array &keeping_cond, const std::variant<py::float_, af::array> &b) {
              auto result = a.copy();
              if (auto pinfo = std::get_if<py::float_>(&b))
                  af::replace(result, keeping_cond, (double) (*pinfo));
              else
                  af::replace(result, keeping_cond, std::get<af::array>(b));
              return result;
          },
          py::arg("a").none(false),
          py::arg("keeping_condition").none(false),
          py::arg("b").none(false),
          "Replace elements of an array based on a conditional array.  The elements kept will be those"
          "matching the condition (this could be a little bit counterintuitive.");

    m.def("whereInPlace",
          [](af::array &a, const af::array &keeping_cond, const std::variant<py::float_, af::array> &b) {
              if (auto pinfo = std::get_if<py::float_>(&b))
                  af::replace(a, keeping_cond, (double) (*pinfo));
              else
                  af::replace(a, keeping_cond, std::get<af::array>(b));
          },
          py::arg("a").none(false),
          py::arg("keeping_condition").none(false),
          py::arg("b").none(false),
          "Replace elements of an array based on a conditional array.  The elements kept will be those"
          "matching the condition (this could be a little bit counterintuitive.");

    m.def("shift",
          [](const af::array &a, const int x, const int y, const int z, const int w) {
              return af::shift(a, x, y, z, w);
          },
          py::arg("a").none(false),
          py::arg("x").none(false),
          py::arg("y") = 0,
          py::arg("z") = 0,
          py::arg("w") = 0,
          "Shifts data in a circular buffer fashion along a chosen dimension");

    m.def("tile",
          [](const af::array &a, const uint x, const uint y, const uint z, const uint w) {
              return af::tile(a, x, y, z, w);
          },
          py::arg("a").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 1,
          py::arg("w") = 1,
          "Repeats an array along the specified dimension");

    m.def("tile",
          [](const af::array &a, const af::dim4 &dims) {
              return af::tile(a, dims);
          },
          py::arg("a").none(false),
          py::arg("dims").none(false),
          "Repeats an array along the specified dimension");

    m.def("transpose",
          [](const af::array &a, const bool conjugate) {
              return af::transpose(a, conjugate);
          },
          py::arg("a").none(false),
          py::arg("dims") = false,
          "Performs a standard matrix transpose");

    m.def("transposeInPlace",
          [](af::array &a, const bool conjugate) {
              af::transposeInPlace(a, conjugate);
          },
          py::arg("a").none(false),
          py::arg("conjugate") = false,
          "Performs a standard matrix transpose, directly over the existing array");

    m.def("cast",
          [](const af::array &a, const af::dtype &type) {
              return a.as(type);
          },
          py::arg("a").none(false),
          py::arg("type").none(false),
          "Creates a new array by casting the original array");
}
