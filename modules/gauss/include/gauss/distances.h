#ifndef GAUSS_DISTANCES_H
#define GAUSS_DISTANCES_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <optional>

namespace gauss::distances {

typedef struct distance_algorithm {
    bool same_length;
    bool is_symmetric;
    std::optional<af::dtype> resultType;
    std::function<af::array(const af::array&, const af::array&)> compute;
} distance_algorithm_t;

distance_algorithm_t euclidian();
distance_algorithm_t hamming();
distance_algorithm_t manhattan();
distance_algorithm_t chebyshev();
distance_algorithm_t minkowshi(double p);
distance_algorithm_t sbd();
distance_algorithm_t dtw();

af::array compute(const distance_algorithm_t& algo, const af::array& xa);
af::array compute(const distance_algorithm_t& algo, const af::array& xa, const af::array &xb);

}  // namespace gauss

#endif
