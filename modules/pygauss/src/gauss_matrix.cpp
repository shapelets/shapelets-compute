#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;
namespace gmatrix = gauss::matrix;

struct matrix_profile {
    matrix_profile(af::array profile, af::array index): profile(std::move(profile)), index(std::move(index)) {}
    af::array profile;
    af::array index;
};

void gauss_matrix_bindings(py::module &m){

    py::class_<matrix_profile>(m, "MatrixProfile")
            .def(py::init<const af::array&, const af::array&>())
            .def_readwrite("profile", &matrix_profile::profile)
            .def_readwrite("index", &matrix_profile::index);

    m.def("matrixprofile",
          [](const af::array &ta, const py::int_& m, const std::optional<af::array> &tb) {
              af::array profile;
              af::array index;
              if (tb.has_value())
                  gmatrix::matrixProfile(ta, tb.value(), m, profile, index);
              else
                  gmatrix::matrixProfile(ta, m, profile, index);
              return matrix_profile {profile, index};
          },
          py::arg("ta").none(false),
          py::arg("m").none(false),
          py::arg("tb") = py::none(),
          "TODO");

    m.def("matrixprofileLR",
          [](const af::array &ta, const py::int_& m) {
              af::array left_profile;
              af::array left_index;
              af::array right_profile;
              af::array right_index;

              gmatrix::matrixProfileLR(ta, m, left_profile, left_index, right_profile, right_index);

              py::dict result;
              result["left"] = matrix_profile { left_profile, left_index };
              result["right"] = matrix_profile {right_profile, right_index};

              return result;
          },
          py::arg("ta").none(false),
          py::arg("m").none(false),
          "TODO");

}