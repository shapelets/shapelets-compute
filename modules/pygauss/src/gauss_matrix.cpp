#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;
namespace gmatrix = gauss::matrix;


void pygauss::bindings::matrix_profile_functions(py::module_ &m)
{

    m.def(
        "mass",
        [](const py::object& queries, const py::object& series) {
            auto qs = arraylike::as_array_checked(queries);
            arraylike::ensure_floating(qs);
            auto ts = arraylike::as_array_checked(series);
            arraylike::ensure_floating(ts);

            af::array distances;
            gmatrix::mass(qs, ts, distances);
            return distances;        
        },
        py::arg("queries").none(false),
        py::arg("series").none(false),
        "TODO"
    );

    m.def(
        "matrixprofile",
        [](const py::object &series_a, const int64_t m, const std::optional<py::object> &series_b) {
            auto ta = arraylike::as_array_checked(series_a);
            arraylike::ensure_floating(ta);

            af::array profile;
            af::array index;

            if (series_b.has_value()) {
                auto tb = arraylike::as_array_checked(series_b.value());
                arraylike::ensure_floating(tb);
                gmatrix::matrixProfile(ta, tb, m, profile, index);
            }
            else {
                gmatrix::matrixProfile(ta, m, profile, index);
            }

            return py::make_tuple(profile, index);
        },
        py::arg("series_a").none(false),
        py::arg("m").none(false),
        py::arg("series_b") = py::none(),
        "TODO");

    m.def(
        "matrixprofileLR",
        [](const py::object &series_a, const int64_t m) {
            auto ta = arraylike::as_array_checked(series_a);
            arraylike::ensure_floating(ta);

            af::array left_profile;
            af::array left_index;
            af::array right_profile;
            af::array right_index;

            gmatrix::matrixProfileLR(ta, m, left_profile, left_index, right_profile, right_index);

            py::dict result;
            result["left"] = py::make_tuple(left_profile, left_index);
            result["right"] = py::make_tuple(right_profile, right_index);
            return result;
        },
        py::arg("ta").none(false),
        py::arg("m").none(false),
        "TODO");
}
