/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_DIMENSIONALITY_H
#define GAUSS_DIMENSIONALITY_H

#include <arrayfire.h>
#include <gauss/defines.h>

#include <vector>

namespace gauss::dimensionality {

using Point = std::pair<float, float>;

using Segment = std::pair<int, int>;

/**
 * @brief Piecewise Aggregate Approximation (PAA) approximates a time series \f$X\f$ of length \f$n\f$ into vector
 * \f$\bar{X}=(\bar{x}_{1},…,\bar{x}_{M})\f$ of any arbitrary length \f$M \leq n\f$ where each of \f$\bar{x_{i}}\f$ is
 * calculated as follows:
 * \f[
 * \bar{x}_{i} = \frac{M}{n} \sum_{j=n/M(i-1)+1}^{(n/M)i} x_{j}.
 * \f]
 * Which simply means that in order to reduce the dimensionality from \f$n\f$ to \f$M\f$, we first divide the original
 * time series into \f$M\f$ equally sized frames and secondly compute the mean values for each frame. The sequence
 * assembled from the mean values is the PAA approximation (i.e., transform) of the original time series.
 *
 * @param points Set of points.
 * @param bins Sets the total number of divisions.
 *
 * @return result A vector of Points with the reduced dimensionality.
 */
GAUSSAPI std::vector<Point> PAA(const std::vector<Point> &points, int bins);

/**
 * @brief Piecewise Aggregate Approximation (PAA) approximates a time series \f$X\f$ of length \f$n\f$ into vector
 * \f$\bar{X}=(\bar{x}_{1},…,\bar{x}_{M})\f$ of any arbitrary length \f$M \leq n\f$ where each of \f$\bar{x_{i}}\f$ is
 * calculated as follows:
 * \f[
 * \bar{x}_{i} = \frac{M}{n} \sum_{j=n/M(i-1)+1}^{(n/M)i} x_{j}.
 * \f]
 * Which simply means that in order to reduce the dimensionality from \f$n\f$ to \f$M\f$, we first divide the original
 * time series into \f$M\f$ equally sized frames and secondly compute the mean values for each frame. The sequence
 * assembled from the mean values is the PAA approximation (i.e., transform) of the original time series.
 *
 * @param a Set of points.
 * @param bins Sets the total number of divisions.
 *
 * @return af::array An array of points with the reduced dimensionality.
 */
GAUSSAPI af::array PAA(const af::array &a, int bins);

/**
 * @brief Calculates the number of Perceptually Important Points (PIP) in the time series.
 *
 * [1] Fu TC, Chung FL, Luk R, and Ng CM. Representing financial time series based on data point importance.
 * Engineering Applications of Artificial Intelligence, 21(2):277-300, 2008.
 *
 * @param ts Expects an input array whose dimension zero is the length of the time series.
 * @param numberIPs The number of points to be returned.
 *
 * @return af::array Array with the most Perceptually Important numPoints.
 */
GAUSSAPI af::array PIP(const af::array &ts, int numberIPs);

/**
 * @brief Applies the Piecewise Linear Approximation (PLA BottomUP) to the time series.
 *
 * [1] Zhu Y, Wu D, Li Sh (2007). A Piecewise Linear Representation Method of Time Series Based on Feature Points.
 * Knowledge-Based Intelligent Information and Engineering Systems 4693:1066-1072.
 *
 * @param ts Expects an input vector containing the set of points to be reduced.
 * @param maxError The maximum approximation error allowed.
 *
 * @return std::vector Vector with the reduced number of points.
 */
GAUSSAPI std::vector<Point> PLABottomUp(const std::vector<Point> &ts, float maxError);

/**
 * @brief Applies the Piecewise Linear Approximation (PLA BottomUP) to the time series.
 *
 * [1] Zhu Y, Wu D, Li Sh (2007). A Piecewise Linear Representation Method of Time Series Based on Feature Points.
 * Knowledge-Based Intelligent Information and Engineering Systems 4693:1066-1072.
 *
 * @param ts Expects an af::array containing the set of points to be reduced. The first component of the points in
 * the first column and the second component of the points in the second column.
 * @param maxError The maximum approximation error allowed.
 *
 * @return af::array with the reduced number of points.
 */
GAUSSAPI af::array PLABottomUp(const af::array &ts, float maxError);

/**
 * @brief Applies the Piecewise Linear Approximation (PLA Sliding Window) to the time series.
 *
 * [1] Zhu Y, Wu D, Li Sh (2007). A Piecewise Linear Representation Method of Time Series Based on Feature Points.
 * Knowledge-Based Intelligent Information and Engineering Systems 4693:1066-1072.
 *
 * @param ts Expects an input vector containing the set of points to be reduced.
 * @param maxError The maximum approximation error allowed.
 *
 * @return std::vector Vector with the reduced number of points.
 */
GAUSSAPI std::vector<Point> PLASlidingWindow(const std::vector<Point> &ts, float maxError);

/**
 * @brief Applies the Piecewise Linear Approximation (PLA Sliding Window) to the time series.
 *
 * [1] Zhu Y, Wu D, Li Sh (2007). A Piecewise Linear Representation Method of Time Series Based on Feature Points.
 * Knowledge-Based Intelligent Information and Engineering Systems 4693:1066-1072.
 *
 * @param ts Expects an af::array containing the set of points to be reduced. The first component of the points in
 * the first column and the second component of the points in the second column.
 * @param maxError The maximum approximation error allowed.
 *
 * @return af::array with the reduced number of points.
 */
GAUSSAPI af::array PLASlidingWindow(const af::array &ts, float maxError);

/**
 * @brief The Ramer–Douglas–Peucker algorithm (RDP) is an algorithm for reducing the number of points in a curve
 * that is approximated by a series of points. It reduces a set of points depending on the perpendicular distance of the
 * points and epsilon, the greater epsilon, more points are deleted.
 *
 * [1] Urs Ramer, "An iterative procedure for the polygonal approximation of plane curves", Computer Graphics and Image
 * Processing, 1(3), 244–256 (1972) doi:10.1016/S0146-664X(72)80017-0.
 *
 * [2] David Douglas & Thomas Peucker, "Algorithms for the reduction of the number of points required to represent a
 * digitized line or its caricature", The Canadian Cartographer 10(2), 112–122 (1973) doi:10.3138/FM57-6770-U75U-7727
 *
 * @param pointList Set of input points.
 * @param epsilon It acts as the threshold value to decide which points should be considered meaningful or not.
 *
 * @return std:vector<algos::dimensionality::Point> with the selected points.
 */
GAUSSAPI std::vector<Point> ramerDouglasPeucker(const std::vector<Point> &pointList, double epsilon);

/**
 * @brief The Ramer–Douglas–Peucker algorithm (RDP) is an algorithm for reducing the number of points in a curve
 * that is approximated by a series of points. It reduces a set of points depending on the perpendicular distance of the
 * points and epsilon, the greater epsilon, more points are deleted.
 *
 * [1] Urs Ramer, "An iterative procedure for the polygonal approximation of plane curves", Computer Graphics and Image
 * Processing, 1(3), 244–256 (1972) doi:10.1016/S0146-664X(72)80017-0.
 *
 * [2] David Douglas & Thomas Peucker, "Algorithms for the reduction of the number of points required to represent a
 * digitized line or its caricature", The Canadian Cartographer 10(2), 112–122 (1973) doi:10.3138/FM57-6770-U75U-7727
 *
 * @param pointList Set of input points.
 * @param epsilon It acts as the threshold value to decide which points should be considered meaningful or not.
 *
 * @return af::array with the selected points.
 */
GAUSSAPI af::array ramerDouglasPeucker(const af::array &pointList, double epsilon);

/**
 * @brief Symbolic Aggregate approXimation (SAX). It transforms a numeric time series into a time series of symbols with
 * the same size. The algorithm was proposed by Lin et al.) and extends the PAA-based approach inheriting the original
 * algorithm simplicity and low computational complexity while providing satisfactory sensitivity and selectivity in
 * range query processing. Moreover, the use of a symbolic representation opened a door to the existing wealth of
 * data-structures and string-manipulation algorithms in computer science such as hashing, regular expression, pattern
 * matching, suffix trees, and grammatical inference.
 *
 * [1] Lin, J., Keogh, E., Lonardi, S. & Chiu, B. (2003) A Symbolic Representation of Time Series, with Implications for
 * Streaming Algorithms. In proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge
 * Discovery. San Diego, CA. June 13.
 *
 * @param a Array with the input time series.
 * @param alphabetSize Number of element within the alphabet.
 *
 * @return result An array of symbols.
 */
GAUSSAPI af::array SAX(const af::array &a, int alphabetSize);

/**
 * @brief Reduces a set of points by applying the Visvalingam method (minimum triangle area) until the number
 * of points is reduced to numPoints.
 *
 * [1] M. Visvalingam and J. D. Whyatt, Line generalisation by repeated elimination of points,
 * The Cartographic Journal, 1993.
 *
 * @param pointList Expects an input vector of points.
 * @param numPoints Sets the number of points returned after the execution of the method.
 * @param scale Sets the precision used to compute the areas of the triangularization, the longer, the more accurate.
 *
 * @return std:vector<algos::dimensionality::Point> where the number of points has been reduced to numPoints.
 */
GAUSSAPI std::vector<Point> visvalingam(const std::vector<Point> &pointList, int64_t numPoints,
                                        int64_t scale = 1000000000);

/**
 * @brief Reduces a set of points by applying the Visvalingam method (minimum triangle area) until the number
 * of points is reduced to numPoints.
 *
 * [1] M. Visvalingam and J. D. Whyatt, Line generalisation by repeated elimination of points,
 * The Cartographic Journal, 1993.
 *
 * @param pointList Expects an input array formed by to columns where the first column is interpreted as the x cordinate
 * of a point and the second column as the y coordinate.
 * @param numPoints Sets the number of points returned after the execution of the method.
 *
 * @return af::array where the number of points has been reduced to numPoints.
 */
GAUSSAPI af::array visvalingam(const af::array &pointList, int numPoints);

}  // namespace gauss

#endif
