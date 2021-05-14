#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

namespace py = pybind11;


void pygauss::bindings::extract_and_transform_operations(py::module &m) {

    m.def("unpack",
        [](const py::object &array_like, 
           const int wx, const int wy,
           const int sx, const int sy, 
           const int px, const int py,
           const bool is_column) {

            auto a = arraylike::as_array_checked(array_like);
            return af::unwrap(a,wx, wy, sx, sy, px, py, is_column);
        },
        py::arg("array_like").none(false),
        py::arg("wx").none(false),
        py::arg("wy").none(false),
        py::arg("sx").none(false),
        py::arg("sy").none(false),
        py::arg("px") = 0,
        py::arg("py") = 0,
        py::arg("is_column") = true);

    m.def("pack",
        [](const py::object &array_like,
           const int ox, const int oy, const int wx, const int wy, const int sx, const int sy,
           const int px, const int py, const bool is_column) {
            auto a = arraylike::as_array_checked(array_like);
            return af::wrap(a, ox, oy, wx, wy, sx, sy, px, py, is_column);
        },
        py::arg("array_like").none(false),
        py::arg("ox").none(false),
        py::arg("oy").none(false),
        py::arg("wx").none(false),
        py::arg("wy").none(false),
        py::arg("sx").none(false),
        py::arg("sy").none(false),
        py::arg("px") = 0,
        py::arg("py") = 0,
        py::arg("is_column") = true);

    m.def("pad",
          [](const py::object& array_like, const af::dim4 &begin, const af::dim4 &end, const af::borderType fill_type) {
              auto a = arraylike::as_array_checked(array_like);
              return af::pad(a, begin, end, fill_type);
          },
          py::arg("array_like").none(false),
          py::arg("begin").none(false),
          py::arg("end").none(false),
          py::arg("fill_type").none(false));

    m.def("lower",
          [](const py::object& array_like, bool unit_diag) {
              auto a = arraylike::as_array_checked(array_like);
              return af::lower(a, unit_diag);
          },
          py::arg("array_like").none(false),
          py::arg("unit_diag") = false);

    m.def("upper",
          [](const py::object& array_like, bool unit_diag) {
              auto a = arraylike::as_array_checked(array_like);
              return af::upper(a, unit_diag);
          },
          py::arg("array_like").none(false),
          py::arg("unit_diag") = false);

    m.def("reshape",
          [](const py::object& array_like, const af::dim4 &shape) {
              auto a = arraylike::as_array_checked(array_like);
              return af::moddims(a, shape);
          },
          py::arg("array_like").none(false),
          py::arg("shape").none(false));

    m.def("flat",
          [](const py::object& array_like) {
              auto a = arraylike::as_array_checked(array_like);
              return af::flat(a);
          },
          py::arg("array_like").none(false));

    m.def("flip",
          [](const py::object& array_like, const uint32_t dimension) {
              auto a = arraylike::as_array_checked(array_like);
              return af::flip(a, dimension);
          },
          py::arg("array_like").none(false),
          py::arg("dimension") = 0);

    m.def("reorder",
          [](const py::object& array_like, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::reorder(a, x, y, z, w);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 2,
          py::arg("w") = 3);

    m.def("shift",
          [](const py::object& array_like, const int x, const int y, const int z, const int w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::shift(a, x, y, z, w);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 0,
          py::arg("z") = 0,
          py::arg("w") = 0);

    m.def("tile",
          [](const py::object& array_like, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) {
              auto a = arraylike::as_array_checked(array_like);
              return af::tile(a, x, y, z, w);
          },
          py::arg("array_like").none(false),
          py::arg("x").none(false),
          py::arg("y") = 1,
          py::arg("z") = 1,
          py::arg("w") = 1);

    m.def("tile",
          [](const py::object& array_like, const af::dim4 &dims) {
              auto a = arraylike::as_array_checked(array_like);
              return af::tile(a, dims);
          },
          py::arg("array_like").none(false),
          py::arg("dims").none(false));

    m.def("transpose",
          [](const py::object& array_like, const bool conjugate) {
              auto a = arraylike::as_array_checked(array_like);
              return af::transpose(a, conjugate);
          },
          py::arg("array_like").none(false),
          py::arg("conjugate") = false);

    m.def("cast",
          [](const py::object& array_like, const af::dtype &dtype) {
              auto a = arraylike::as_array_checked(array_like);
              return a.as(dtype);
          },
          py::arg("array_like").none(false),
          py::arg("dtype").none(false));

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
          py::arg("dimension") = 0);

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
          py::arg("y") = py::none());
}
