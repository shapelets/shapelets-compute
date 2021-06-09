/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_CLUSTERING_H
#define GAUSS_CLUSTERING_H

#include <arrayfire.h>
#include <gauss/defines.h>

#include <vector>

namespace gauss::clustering {

/**
 * @brief Calculates the k-means algorithm.
 *
 * [1] S. Lloyd. 1982. Least squares quantization in PCM. IEEE Transactions on Information Theory, 28, 2,
 * Pages 129-137.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 * @param k The number of means to be computed.
 * @param centroids The resulting means or centroids.
 * @param labels The resulting labels of each time series which is the closest centroid.
 * @param tolerance The error tolerance to stop the computation of the centroids.
 * @param maxIterations The maximum number of iterations allowed.
 */
GAUSSAPI void kMeans(const af::array &tss, int k, af::array &centroids, af::array &labels,
                     float tolerance = 0.0000000001, int maxIterations = 100);

/**
 * @brief Calculates the k-shape algorithm.
 *
 * [1] John Paparrizos and Luis Gravano. 2016. k-Shape: Efficient and Accurate Clustering of Time Series.
 * SIGMOD Rec. 45, 1 (June 2016), 69-76.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 * 
 * @param k (in) The number of means to be computed.
 * 
 * @param labels (in-out) It serves both as input for training and at the end of the algorithm, it will contain the 
 * calculated labels.
 * 
 * @param centroids (out) The the computed centroids for each group.
 * 
 * @param maxIterations The maximum number of iterations allowed.
 * 
 * @param rnd_labels When no labels are given, this parameter controls if the initial labels are to be generated 
 * from draws of a uniform distribution; when set to false, the initial labels will be assigned sequentially.
 */
GAUSSAPI void kshape_calibrate(const af::array &tss, int k, af::array &labels, af::array &centroids, 
    const int maxIterations = 100, const bool rnd_labels = false);

/**
 * @brief Classifies the time series as per the centroids computed in the calibration phase.
 * 
 * @param tss Input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 * 
 * @param centroids Input (columnar) array with the centroids computed during a calibration phase.
 * 
 * @returns Labels for each input time series.  The label is a numerical value that matches the
 * column of the centroid.
 */
GAUSSAPI af::array kshape_classify(const af::array &tss, const af::array &centroids);

}  // namespace gauss

#endif
