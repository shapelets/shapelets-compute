#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;
namespace gdist = gauss::distances;

// "euclidian", "hamming", "manhattan", "sbd", "dtw", "mpdist"

typedef enum {
    EUCLIDIAN  = 0,
    HAMMING  = 1,
    MANHATTAN  = 2,
    SBD  = 3,
    DTW = 4,
    MPDIST = 5
} distance_types;

void pygauss::bindings::gauss_distance_functions(py::module_ &m) {

  py::enum_<distance_types>(m, "DistanceType", "Distance Type")
        .value("Euclidian", distance_types::EUCLIDIAN, "")
        .value("Hamming", distance_types::HAMMING, "")
        .value("Manhattan", distance_types::MANHATTAN, "")
        .value("SBD", distance_types::SBD, "")
        .value("DTW", distance_types::DTW, "")
        .value("MPD", distance_types::MPDIST, "")
        .export_values();


  m.def("pdist",
    [](const py::object& array_like, const distance_types dst) {

    },
    py::arg("array_like").none(false),
    py::arg("dst").none(false));

  m.def("cdist",
    [](const py::object& xa, const py::object& xb, const distance_types dst) {

    },
    py::arg("xb").none(false),
    py::arg("xa").none(false),
    py::arg("dst").none(false));

}