#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

#include <utility>

namespace py = pybind11;
namespace gdist = gauss::distances;

typedef enum {
    Additive_Symm_Chi, 
    Avg_L1_Linf, 
    Bhattacharyya, 
    Canberra, 
    Chebyshev, 
    Clark, 
    Cosine, 
    Czekanowski, 
    Dice, 
    Divergence, 
    DTW, 
    Euclidean, 
    Fidelity, 
    Gower, 
    Hamming, 
    Harmonic_mean, 
    Hellinger, 
    Innerproduct, 
    Intersection, 
    Jaccard, 
    Jeffrey, 
    Jensen_Difference, 
    Jensen_Shannon, 
    K_Divergence, 
    Kulczynski, 
    Kullback, 
    Kumar_Johnson, 
    Kumar_Hassebrook, 
    Lorentzian, 
    Manhattan, 
    Matusita, 
    Max_Symmetric_Chi, 
    Min_Symmetric_Chi, 
    Minkowski, 
    MPDist, 
    Neyman, 
    Pearson, 
    Prob_Symmetric_Chi, 
    SBD, 
    Soergel, 
    Sorensen, 
    Square_Chord, 
    Squared_Chi, 
    Squared_Euclidean, 
    Taneja, 
    Topsoe, 
    Vicis_Wave_Hedges, 
    Wave_Hedges
} distance_types;



gauss::distances::distance_algorithm_t enumToAlgo(distance_types dst,py::kwargs kwargs) {
  switch(dst) {
    case distance_types::Additive_Symm_Chi:
          return gauss::distances::additive_symm_chi();
    case distance_types::Avg_L1_Linf:
          return gauss::distances::avg_l1_linf();
    case distance_types::Bhattacharyya:
          return gauss::distances::bhattacharyya();
    case distance_types::Canberra:
          return gauss::distances::canberra();
    case distance_types::Chebyshev:
          return gauss::distances::chebyshev();
    case distance_types::Clark:
          return gauss::distances::clark();
    case distance_types::Cosine:
          return gauss::distances::cosine();
    case distance_types::Czekanowski:
          return gauss::distances::czekanowski();
    case distance_types::Dice:
          return gauss::distances::dice();
    case distance_types::Divergence:
          return gauss::distances::divergence();
    case distance_types::DTW:
          return gauss::distances::dtw();
    case distance_types::Euclidean:
          return gauss::distances::euclidean();
    case distance_types::Fidelity:
          return gauss::distances::fidelity();
    case distance_types::Gower:
          return gauss::distances::gower();
    case distance_types::Hamming:
          return gauss::distances::hamming();
    case distance_types::Harmonic_mean:
          return gauss::distances::harmonic_mean();
    case distance_types::Hellinger:
          return gauss::distances::hellinger();
    case distance_types::Innerproduct:
          return gauss::distances::innerproduct();
    case distance_types::Intersection:
          return gauss::distances::intersection();
    case distance_types::Jaccard:
          return gauss::distances::jaccard();
    case distance_types::Jeffrey:
          return gauss::distances::jeffrey();
    case distance_types::Jensen_Difference:
          return gauss::distances::jensen_difference();
    case distance_types::Jensen_Shannon:
          return gauss::distances::jensen_shannon();
    case distance_types::K_Divergence:
          return gauss::distances::k_divergence();
    case distance_types::Kulczynski:
          return gauss::distances::kulczynski();
    case distance_types::Kullback:
          return gauss::distances::kullback();
    case distance_types::Kumar_Johnson:
          return gauss::distances::kumar_johnson();
    case distance_types::Kumar_Hassebrook:
          return gauss::distances::kumarhassebrook();
    case distance_types::Lorentzian:
          return gauss::distances::lorentzian();
    case distance_types::Manhattan:
          return gauss::distances::manhattan();
    case distance_types::Matusita:
          return gauss::distances::matusita();
    case distance_types::Max_Symmetric_Chi:
          return gauss::distances::max_symmetric_chi();
    case distance_types::Min_Symmetric_Chi:
          return gauss::distances::min_symmetric_chi();
    case distance_types::Minkowski:
          {
          auto key = py::str("p");
          if (!kwargs || !kwargs.contains(key)) throw std::invalid_argument("Minkowski requires parameter p");
          auto p = kwargs[key].cast<double>();
          return gauss::distances::minkowski(p);
        }
    case distance_types::MPDist: {
            auto key = py::str("w");
            if (!kwargs || !kwargs.contains(key)) throw std::invalid_argument("MPDist requires parameter w");
            auto w = kwargs[key].cast<int32_t>();
            auto thresKey = py::str("threshold");
            auto threshold = 0.05;
            if (kwargs.contains(thresKey)) {
              threshold = kwargs[thresKey].cast<double>();
            }
            return gauss::distances::mpdist(w, threshold);
          }
    case distance_types::Neyman:
          return gauss::distances::neyman();
    case distance_types::Pearson:
          return gauss::distances::pearson();
    case distance_types::Prob_Symmetric_Chi:
          return gauss::distances::prob_symmetric_chi();
    case distance_types::SBD:
          return gauss::distances::sbd();
    case distance_types::Soergel:
          return gauss::distances::soergel();
    case distance_types::Sorensen:
          return gauss::distances::sorensen();
    case distance_types::Square_Chord:
          return gauss::distances::square_chord();
    case distance_types::Squared_Chi:
          return gauss::distances::squared_chi();
    case distance_types::Squared_Euclidean:
          return gauss::distances::squared_euclidean();
    case distance_types::Taneja:
          return gauss::distances::taneja();
    case distance_types::Topsoe:
          return gauss::distances::topsoe();
    case distance_types::Vicis_Wave_Hedges:
          return gauss::distances::vicis_wave_hedges();
    case distance_types::Wave_Hedges:
          return gauss::distances::wavehedges();
    default:
      throw std::runtime_error("TODO");
  }
}

void pygauss::bindings::gauss_distance_functions(py::module &m) {

  py::enum_<distance_types>(m, "DistanceType", "Distance Type")
        .value("Additive_Symm_Chi", distance_types::Additive_Symm_Chi, "")
        .value("Avg_L1_Linf", distance_types::Avg_L1_Linf, "")
        .value("Bhattacharyya", distance_types::Bhattacharyya, "")
        .value("Canberra", distance_types::Canberra, "")
        .value("Chebyshev", distance_types::Chebyshev, "")
        .value("Clark", distance_types::Clark, "")
        .value("Cosine", distance_types::Cosine, "")
        .value("Czekanowski", distance_types::Czekanowski, "")
        .value("Dice", distance_types::Dice, "")
        .value("Divergence", distance_types::Divergence, "")
        .value("DTW", distance_types::DTW, "")
        .value("Euclidean", distance_types::Euclidean, "")
        .value("Fidelity", distance_types::Fidelity, "")
        .value("Gower", distance_types::Gower, "")
        .value("Hamming", distance_types::Hamming, "")
        .value("Harmonic_mean", distance_types::Harmonic_mean, "")
        .value("Hellinger", distance_types::Hellinger, "")
        .value("Innerproduct", distance_types::Innerproduct, "")
        .value("Intersection", distance_types::Intersection, "")
        .value("Jaccard", distance_types::Jaccard, "")
        .value("Jeffrey", distance_types::Jeffrey, "")
        .value("Jensen_Difference", distance_types::Jensen_Difference, "")
        .value("Jensen_Shannon", distance_types::Jensen_Shannon, "")
        .value("K_Divergence", distance_types::K_Divergence, "")
        .value("Kulczynski", distance_types::Kulczynski, "")
        .value("Kullback", distance_types::Kullback, "")
        .value("Kumar_Johnson", distance_types::Kumar_Johnson, "")
        .value("Kumar_Hassebrook", distance_types::Kumar_Hassebrook, "")
        .value("Lorentzian", distance_types::Lorentzian, "")
        .value("Manhattan", distance_types::Manhattan, "")
        .value("Matusita", distance_types::Matusita, "")
        .value("Max_Symmetric_Chi", distance_types::Max_Symmetric_Chi, "")
        .value("Min_Symmetric_Chi", distance_types::Min_Symmetric_Chi, "")
        .value("Minkowski", distance_types::Minkowski, "")
        .value("MPDist", distance_types::MPDist, "")
        .value("Neyman", distance_types::Neyman, "")
        .value("Pearson", distance_types::Pearson, "")
        .value("Prob_Symmetric_Chi", distance_types::Prob_Symmetric_Chi, "")
        .value("SBD", distance_types::SBD, "")
        .value("Soergel", distance_types::Soergel, "")
        .value("Sorensen", distance_types::Sorensen, "")
        .value("Square_Chord", distance_types::Square_Chord, "")
        .value("Squared_Chi", distance_types::Squared_Chi, "")
        .value("Squared_Euclidean", distance_types::Squared_Euclidean, "")
        .value("Taneja", distance_types::Taneja, "")
        .value("Topsoe", distance_types::Topsoe, "")
        .value("Vicis_Wave_Hedges", distance_types::Vicis_Wave_Hedges, "")
        .value("Wave_Hedges", distance_types::Wave_Hedges, "")
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
    py::arg("xa").none(false),
    py::arg("xb").none(false),
    py::arg("dst").none(false)
    );

}