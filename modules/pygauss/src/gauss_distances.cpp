#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;
namespace gdist = gauss::distances;

typedef enum {
    EUCLIDIAN  = 0,
    HAMMING  = 1,
    MANHATTAN  = 2,
    CHEBYSHEV = 3,
    MINKOWSHI = 4,
    SBD  = 5,
    DTW = 6,
    MPDIST = 7
} distance_types;


gauss::distances::distance_algorithm_t enumToAlgo(distance_types dst,py::kwargs kwargs) {
  switch(dst) {
    case distance_types::EUCLIDIAN:
      return gauss::distances::euclidian();
    case distance_types::HAMMING:
      return gauss::distances::hamming();
    case distance_types::MANHATTAN:
      return gauss::distances::manhattan();
    case distance_types::CHEBYSHEV:
      return gauss::distances::chebyshev();
    case distance_types::MINKOWSHI: {
      auto key = py::str("p");
      if (!kwargs || !kwargs.contains(key)) throw std::invalid_argument("Minkowshi requires parameter p");
      auto p = kwargs[key].cast<double>();
      return gauss::distances::minkowshi(p);
    }
    case distance_types::SBD:
      return gauss::distances::sbd();      
    case distance_types::DTW:
      return gauss::distances::dtw();
    default:
      throw std::runtime_error("TODO");
  }
}

void pygauss::bindings::gauss_distance_functions(py::module_ &m) {

  py::enum_<distance_types>(m, "DistanceType", "Distance Type")
        .value("Euclidian", distance_types::EUCLIDIAN, "")
        .value("Hamming", distance_types::HAMMING, "")
        .value("Manhattan", distance_types::MANHATTAN, "")
        .value("Chebyshev", distance_types::CHEBYSHEV, "")
        .value("Minkowshi", distance_types::MINKOWSHI, "")
        .value("SBD", distance_types::SBD, "")
        .value("DTW", distance_types::DTW, "")
        .value("MPD", distance_types::MPDIST, "")
        .export_values();


  m.def("pdist",
    [](const py::object& array_like, const distance_types distType, py::kwargs kwargs) {
        auto data = arraylike::as_array_checked(array_like);
        return gauss::distances::compute(enumToAlgo(distType, kwargs), data);
    },
    py::arg("array_like").none(false),
    py::arg("distType").none(false)
    );

  m.def("cdist",
    [](const py::object& xa, const py::object& xb, const distance_types distType, py::kwargs kwargs) {
        auto left = arraylike::as_array_checked(xa);
        auto right = arraylike::as_array_checked(xb);
        return gauss::distances::compute(enumToAlgo(distType, kwargs), left, right);
    },
    py::arg("xb").none(false),
    py::arg("xa").none(false),
    py::arg("dst").none(false)
    );

}