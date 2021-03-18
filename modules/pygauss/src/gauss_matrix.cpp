#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <utility>

namespace py = pybind11;
namespace gmatrix = gauss::matrix;


void pygauss::bindings::matrix_profile_functions(py::module &m)
{
    py::class_<gmatrix::snippet_t>(m, "Snippet")
    .def_property_readonly("indices", [](const gmatrix::snippet_t& self) { return self.indices; })
    .def_property_readonly("index", [](const gmatrix::snippet_t& self) { return self.index; })
    .def_property_readonly("distances", [](const gmatrix::snippet_t& self) { return self.distances; })
    .def_property_readonly("pct", [](const gmatrix::snippet_t& self) { return self.pct; })
    .def_property_readonly("window", [](const gmatrix::snippet_t& self) { return self.window; })
    .def_property_readonly("size", [](const gmatrix::snippet_t& self) { return self.size; })
    .def("__repr__",
        [](const gmatrix::snippet_t& self) {
            std::stringstream result;
            auto start_pos = self.index * self.size;
            auto end_pos = start_pos + self.size - 1;
            result << "Snippet [" << start_pos << "," << end_pos << "] (" << std::setprecision(3) << self.pct * 100 << " %)";
            return result.str();
        });

    m.def("cac",
        [](const py::object& profile, const py::object& index, const uint32_t window_size) {
            auto p = arraylike::as_array_checked(profile);
            auto i = arraylike::as_array_checked(index);
            return gmatrix::cac(p, i, window_size);
        },
        py::arg("profile").none(false),
        py::arg("index").none(false),
        py::arg("window_size").none(false));

    m.def("snippets",
         [](const py::object& tsa, const uint32_t snippet_size, const uint32_t num_snippets, std::optional<uint32_t> window_size) {
            auto a = arraylike::as_array_checked(tsa);
            return gmatrix::snippets(a, snippet_size, num_snippets, window_size);
         },
         py::arg("tsa").none(false),
         py::arg("snippet_size").none(false),
         py::arg("num_snippets").none(false),
         py::arg("window_size") = py::none()
         );

    m.def("mpdist_vect", 
        [](const py::object& tsa, const py::object& tsb, long w, std::optional<double> threshold) {
            auto a = arraylike::as_array_checked(tsa);
            auto b = arraylike::as_array_checked(tsb);
            
            if (threshold.has_value()) {
                return gmatrix::mpdist_vector(a, b, w, threshold.value());
            }

            return gmatrix::mpdist_vector(a, b, w);
        },
        py::arg("tsa").none(false),
        py::arg("tsb").none(false),
        py::arg("w").none(false),
        py::arg("threshold") = 0.05
        );

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
