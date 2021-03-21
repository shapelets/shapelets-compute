#ifndef GAUSS_DISTANCES_H
#define GAUSS_DISTANCES_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <optional>
#include <functional>

namespace gauss::distances {

typedef struct distance_algorithm {
    bool same_length;
    bool is_symmetric;
    std::optional<af::dtype> resultType;
    std::function<af::array(const af::array&, const af::array&)> compute;
} distance_algorithm_t;

distance_algorithm_t abs_euclidean();
distance_algorithm_t additive_symm_chi();
distance_algorithm_t avg_l1_linf();
distance_algorithm_t bhattacharyya();
distance_algorithm_t canberra();
distance_algorithm_t chebyshe();
distance_algorithm_t chebyshev();
distance_algorithm_t clark();
distance_algorithm_t cosine();
distance_algorithm_t czekanowski();
distance_algorithm_t dice();
distance_algorithm_t divergence();
distance_algorithm_t dtw();
distance_algorithm_t euclidean();
distance_algorithm_t fidelity();
distance_algorithm_t gower();
distance_algorithm_t hamming();
distance_algorithm_t harmonic_mean();
distance_algorithm_t hellinger();
distance_algorithm_t innerproduct();
distance_algorithm_t intersection();
distance_algorithm_t jaccard();
distance_algorithm_t jensen_shannon();
distance_algorithm_t jeffrey();
distance_algorithm_t jensen_difference();
distance_algorithm_t k_divergence();
distance_algorithm_t kulczynski();
distance_algorithm_t kullback();
distance_algorithm_t kumar_johnson();
distance_algorithm_t kumarhassebrook();
distance_algorithm_t lorentzian();
distance_algorithm_t manhattan();
distance_algorithm_t matusita();
distance_algorithm_t max_symmetric_chi();
distance_algorithm_t min_symmetric_chi();
distance_algorithm_t minkowshi(double p);
distance_algorithm_t mpdist(int32_t w, double threshold = 0.05);
distance_algorithm_t neyman();
distance_algorithm_t pearson();
distance_algorithm_t prob_symmetric_chi();
distance_algorithm_t sbd();
distance_algorithm_t soergel();
distance_algorithm_t sorensen();
distance_algorithm_t square_chord();
distance_algorithm_t squared_chi();
distance_algorithm_t squared_euclidean();
distance_algorithm_t taneja();
distance_algorithm_t topsoe();
distance_algorithm_t vicis_wave_hedges();
distance_algorithm_t wavehedges();

af::array compute(const distance_algorithm_t& algo, const af::array& xa);
af::array compute(const distance_algorithm_t& algo, const af::array& xa, const af::array &xb);

}  // namespace gauss

#endif
