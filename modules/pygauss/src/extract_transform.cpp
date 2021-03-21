#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

namespace py = pybind11;


void pygauss::bindings::extract_and_transform_operations(py::module &m) {

    m.def("pad",
          [](const py::object& array_like, const af::dim4 &begin, const af::dim4 &end, const af::borderType fill_type) {
              auto a = arraylike::as_array_checked(array_like);
              return af::pad(a, begin, end, fill_type);
          },
          py::arg("array_like").none(false),
          py::arg("begin").none(false),
          py::arg("end").none(false),
          py::arg("fill_type").none(false),
          R"__( Pads an array

          Ensure the tuples for begin and end are specific as they represent the increase at the beginning and
          the ending; if you want to add one row at the beginning and one row to the end use (1,0,0,0) and
          (1,0,0,0) as parameters.

          )__");

    m.def("lower",
          [](const py::object& array_like, bool unit_diag) {
              auto a = arraylike::as_array_checked(array_like);
              return af::lower(a, unit_diag);
          },
          py::arg("array_like").none(false),
          py::arg("unit_diag") = false,
          "Create a lower triangular matrix from input array\n"
          "The parameter unit_diag forces the diagonal elements to be one.");

    m.def("upper",
          [](const py::object& array_like, bool unit_diag) {
              auto a = arraylike::as_array_checked(array_like);
              return af::upper(a, unit_diag);
          },
          py::arg("array_like").none(false),
          py::arg("unit_diag") = false,
          "Create a upper triangular matrix from input array\n"
          "The parameter unit_diag forces the diagonal elements to be one.");

    m.def("reshape",
          [](const py::object& array_like, const af::dim4 &shape) {
              auto a = arraylike::as_array_checked(array_like);
              return af::moddims(a, shape);
          },
          py::arg("array_like").none(false),
          py::arg("shape").none(false),
          "Changes the dimensions of an array without changing the data");

    m.def("flat",
          [](const py::object& array_like) {
              auto a = arraylike::as_array_checked(array_like);
              return af::flat(a);
          },
          py::arg("array_like").none(false),
          "It flattens an array to one dimension");

    m.def("flip",
          [](const py::object& array_like, const uint32_t dimension) {
              auto a = arraylike::as_array_checked(array_like);
              return af::flip(a, dimension);
          },
          py::arg("array_like").none(false),
          py::arg("dimension") = 0,
          "Flips an array along a dimension");

    m.def("reorder",
          [](const py::object& array_like, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::reorder(a, x, y, z);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 2,
          py::arg("w") = 3,
          "It modifies the order of data within an array by exchanging data according to the change "
          "in dimensionality. The linear ordering of data within the array is preserved.");

    m.def("shift",
          [](const py::object& array_like, const int x, const int y, const int z, const int w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::shift(a, x, y, z, w);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 0,
          py::arg("z") = 0,
          py::arg("w") = 0,
          "Shifts data in a circular buffer fashion along a chosen dimension");

    m.def("tile",
          [](const py::object& array_like, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::tile(a, x, y, z, w);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 1,
          py::arg("w") = 1,
          "Repeats an array along the specified dimension");

    m.def("tile",
          [](const py::object& array_like, const af::dim4 &dims) {
              auto a = arraylike::as_array_checked(array_like);
              return af::tile(a, dims);
          },
          py::arg("array_like").none(false),
          py::arg("dims").none(false),
          "Repeats an array along the specified dimension");

    m.def("transpose",
          [](const py::object& array_like, const bool conjugate) {
              auto a = arraylike::as_array_checked(array_like);
              return af::transpose(a, conjugate);
          },
          py::arg("array_like").none(false),
          py::arg("dims") = false,
          "Performs a standard matrix transpose");

    m.def("cast",
          [](const py::object& array_like, const af::dtype &dtype) {
              auto a = arraylike::as_array_checked(array_like);
              return a.as(dtype);
          },
          py::arg("array_like").none(false),
          py::arg("dtype").none(false),
          "Creates a new array by casting the original array");

    m.def("join",
          [](const py::list& lst, const int dimension) {

              auto arr_objs = std::vector<af::array>();
              for (auto entry: lst) {
                  auto obj = entry.cast<py::object>();
                  auto converted = arraylike::as_array_checked(obj);
                  arr_objs.push_back(converted);
              }

              af::array acc = arr_objs[0];
              auto i = 1;
              while (i < arr_objs.size()) {
                  auto left = arr_objs.size() - i;
                  if (left >= 3) {
                      acc = af::join(dimension, acc, arr_objs[i], arr_objs[i+1], arr_objs[i+2]);
                      i += 3;
                  } else if (left >= 2) {
                      acc = af::join(dimension, acc, arr_objs[i], arr_objs[i+1]);
                      i += 2;
                  } else {
                      acc = af::join(dimension, acc, arr_objs[i]);
                      i += 1;
                  }
              }

              return acc;
          },
          py::arg("lst").none(false),
          py::arg("dimension") = 0,
          R"_(
    Joins any number of arrays along a particular dimension.

    In the case that not all objects in the lst are arrays, the parameters shape and dtype would guide
    the transformation; if those parameters are not set, the first array in the list will determine
    the shape and type of those entries that are not defined as arrays.
    )_");

    m.def("where",
          [](const py::object &condition, const py::object &x, const py::object &y) {
              auto c = arraylike::as_array_checked(condition);
              if (x.is_none() || y.is_none()) {
                  return af::iszero(c);
              }

              auto conversion = arraylike::as_array(x, y);
              if (!conversion)
                  throw std::invalid_argument("Unable to convert to array");

              auto [s, r] = conversion.value();
              auto output = s.copy();

              //replace needs identical types.
              if (s.type() != r.type())
                output = output.as(r.type());

              af::replace(output, c, r);
              return output;
          },
          py::arg("condition").none(false),
          py::arg("x") = py::none(),
          py::arg("y") = py::none(),
          "An array with elements from x where condition is True, and elements from y elsewhere.");

//    m.def("whereInPlace",
//          [](af::array &a, const af::array &keeping_cond, const std::variant<py::float_, af::array> &b) {
//              if (auto pinfo = std::get_if<py::float_>(&b))
//                  af::replace(a, keeping_cond, (double) (*pinfo));
//              else
//                  af::replace(a, keeping_cond, std::get<af::array>(b));
//          },
//          py::arg("a").none(false),
//          py::arg("keeping_condition").none(false),
//          py::arg("b").none(false),
//          "Replace elements of an array based on a conditional array.  The elements kept will be those"
//          "matching the condition (this could be a little bit counterintuitive.");
//
//    m.def("transposeInPlace",
//          [](const py::object& array_like, const bool conjugate) {
//              auto a = arraylike::cast(array_like);
//              af::transposeInPlace(a, conjugate);
//          },
//          py::arg("a").none(false),
//          py::arg("conjugate") = false,
//          "Performs a standard matrix transpose, directly over the existing array");

}
