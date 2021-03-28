#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;

void pygauss::bindings::gauss_dimensionality_functions(py::module &m) {

    m.def(
        "visvalingam",
        [](const py::object &x, const py::object &y, const int num_points){
            auto xx = arraylike::as_array_checked(x);
            arraylike::ensure_floating(xx);
            auto yy = arraylike::as_array_checked(y);
            arraylike::ensure_floating(yy);
            return gauss::dimensionality::visvalingam(af::join(1, xx, yy), num_points);
        },
        py::arg("x").none(false),
        py::arg("y").none(false),
        py::arg("num_points").none(false));

    // it is broken
    // m.def(
    //     "sax",
    //     [](const py::object &data, const int alphabet_size){
    //         auto ts = arraylike::as_array_checked(data);
    //         arraylike::ensure_floating(ts);
    //         return gauss::dimensionality::SAX(ts, alphabet_size);
    //     },
    //     py::arg("data").none(false),
    //     py::arg("alphabet_size").none(false));

    m.def(
        "paa",
        [](const py::object &x, const py::object &y, const int bins) {
            auto xx = arraylike::as_array_checked(x);
            arraylike::ensure_floating(xx);
            auto yy = arraylike::as_array_checked(y);
            arraylike::ensure_floating(yy);
            return gauss::dimensionality::PAA(af::join(1, xx, yy), bins);
        },
        py::arg("x").none(false),
        py::arg("y").none(false),
        py::arg("bins").none(false));

    m.def(
        "pip",
        [](const py::object &x, const py::object &y, const int ips) {
            auto xx = arraylike::as_array_checked(x);
            arraylike::ensure_floating(xx);
            auto yy = arraylike::as_array_checked(y);
            arraylike::ensure_floating(yy);
            return gauss::dimensionality::PIP(af::join(1, xx, yy), ips);
        },
        py::arg("x").none(false),
        py::arg("y").none(false),
        py::arg("ips").none(false));  

    // m.def(
    //     "pla_bottom_up",
    //     [](const py::object &x, const py::object &y, const float max_error) {
    //         auto xx = arraylike::as_array_checked(x);
    //         arraylike::ensure_floating(xx);
    //         auto yy = arraylike::as_array_checked(y);
    //         arraylike::ensure_floating(yy);
    //         return gauss::dimensionality::PLABottomUp(af::join(1, xx, yy), max_error);
    //     },
    //     py::arg("x").none(false),
    //     py::arg("y").none(false),
    //     py::arg("max_error").none(false));

    // m.def(
    //     "pla_sliding",
    //     [](const py::object &x, const py::object &y, const float max_error) {
    //         auto xx = arraylike::as_array_checked(x);
    //         arraylike::ensure_floating(xx);
    //         auto yy = arraylike::as_array_checked(y);
    //         arraylike::ensure_floating(yy);
    //         return gauss::dimensionality::PLASlidingWindow(af::join(1, xx, yy), max_error);
    //     },
    //     py::arg("x").none(false),
    //     py::arg("y").none(false),
    //     py::arg("max_error").none(false));

}


