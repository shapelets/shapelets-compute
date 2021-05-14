#ifndef GAUSS_DISTANCES_H
#define GAUSS_DISTANCES_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <optional>
#include <functional>

namespace gauss::distances {

/**
 * Structure that describes a distance / similitude algorithm
 */ 
typedef struct distance_algorithm {
    // Requires all column vectors to be of the same length
    bool same_length;

    // Informs of possible optimisations when d_ij=d_ji
    bool is_symmetric;
    
    // Optionally describes the preferred result type of 
    // the computation.  This is used to allocate the 
    // result matrix in compute (see below) public 
    // methods
    std::optional<af::dtype> resultType;
    
    // Computes the distance of the column vector in the 
    // first parameters against all column vectors in the 
    // second argument.
    std::function<af::array(const af::array&, const af::array&)> compute;

} distance_algorithm_t;

/**
 * @brief Runs the algo for every column in xa to all the others.
 * if the algorithm is symmetric, it will only do half of the work
 */ 
af::array compute(const distance_algorithm_t& algo, const af::array& xa);

/**
 * @brief Runs algo for every column in xa against all columns in xb
 */ 
af::array compute(const distance_algorithm_t& algo, const af::array& xa, const af::array &xb);


/////////////////
// Built-in Algos
/////////////////

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
distance_algorithm_t minkowski(double p);
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
distance_algorithm_t tanimoto();
distance_algorithm_t ruzicka();
distance_algorithm_t motyka();

}  // namespace gauss

#endif
