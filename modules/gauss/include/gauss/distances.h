#ifndef GAUSS_DISTANCES_H
#define GAUSS_DISTANCES_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::distances {

/**
 * @brief Calculates the Dynamic Time Warping Distance.
 *
 * @param a The input time series of reference.
 * @param b The input query.
 *
 * @return array The resulting distance between a and b.
 */
GAUSSAPI double dtw(const std::vector<double> &a, const std::vector<double> &b);

/**
 * @brief Calculates the Dynamic Time Warping Distance.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return af::array An upper triangular matrix where each position corresponds to the distance between
 * two time series. Diagonal elements will be zero. For example: Position row 0 column 1 records the
 * distance between time series 0 and time series 1.
 */
GAUSSAPI af::array dtw(const af::array &tss);

/**
 * @brief Calculates euclidean distances between time series.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return af::array An upper triangular matrix where each position corresponds to the distance between two
 * time series. Diagonal elements will be zero. For example: Position row 0 column 1 records the distance
 * between time series 0 and time series 1.
 */
GAUSSAPI af::array euclidean(const af::array &tss);

/**
 * @brief Calculates hamming distances between time series.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return af::array An upper triangular matrix where each position corresponds to the distance between two
 * time series. Diagonal elements will be zero. For example: Position row 0 column 1 records the distance
 * between time series 0 and time series 1.
 */
GAUSSAPI af::array hamming(const af::array &tss);

/**
 * @brief Calculates manhattan distances between time series.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return af::array An upper triangular matrix where each position corresponds to the distance between two
 * time series. Diagonal elements will be zero. For example: Position row 0 column 1 records the distance
 * between time series 0 and time series 1.
 */
GAUSSAPI af::array manhattan(const af::array &tss);

/**
 * @brief Calculates the Shape-Based distance (SBD). It computes the normalized cross-correlation and it returns 1.0
 * minus the value that maximizes the correlation value between each pair of time series.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return array An upper triangular matrix where each position corresponds to the distance between two time series.
 * Diagonal elements will be zero. For example: Position row 0 column 1 records the distance between time series 0
 * and time series 1.
 */
GAUSSAPI af::array sbd(const af::array &tss);

/**
 * @brief Calculates non squared version of the euclidean distance.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return array An upper triangular matrix where each position corresponds to the distance between two time series.
 * Diagonal elements will be zero. For example: Position row 0 column 1 records the distance between time series 0
 * and time series 1.
 */
GAUSSAPI af::array squaredEuclidean(const af::array &tss);

}  // namespace gauss

#endif
