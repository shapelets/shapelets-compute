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
 * @param k The number of means to be computed.
 * @param centroids The resulting means or centroids.
 * @param labels The resulting labels of each time series which is the closest centroid.
 * @param tolerance The error tolerance to stop the computation of the centroids.
 * @param maxIterations The maximum number of iterations allowed.
 */
GAUSSAPI void kShape(const af::array &tss, int k, af::array &centroids, af::array &labels,
                     float tolerance = 0.0000000001, int maxIterations = 100);

}  // namespace gauss

#endif