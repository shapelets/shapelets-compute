#ifndef GAUSS_NORMALIZATION_H
#define GAUSS_NORMALIZATION_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::normalization {

/**
 * @brief Removes the mean from the input series
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 */  
GAUSSAPI af::array detrend(const af::array &tss) ;

/**
 * @brief Normalizes the given time series according to its maximum value and adjusts each value within the range
 * (-1, 1).
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 *
 * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0) have been
 * normalized by dividing each number by 10^j, where j is the number of integer digits of the max number in the time
 * series.
 */
GAUSSAPI af::array decimalScalingNorm(const af::array &tss);

/**
 * @brief Same as decimalScalingNorm, but it performs the operation in place, without allocating further memory.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 */
GAUSSAPI void decimalScalingNormInPlace(af::array &tss);

/**
 * @brief Normalizes the given time series according to its minimum and maximum value and adjusts each value within the
 * range [low, high].
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param high Maximum final value (Defaults to 1.0).
 * @param low  Minimum final value (Defaults to 0.0).
 * @param epsilon Safeguard for constant (or near constant) time series as the operation implies a unit scale operation
 * between min and max values in the tss.
 *
 * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0) have been
 * normalized by maximum and minimum values, and scaled as per high and low parameters.
 */
GAUSSAPI af::array maxMinNorm(const af::array &tss, double high = 1.0, double low = 0.0, double epsilon = 0.00000001);

/**
 * @brief Same as maxMinNorm, but it performs the operation in place, without allocating further memory.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param high Maximum final value (Defaults to 1.0).
 * @param low  Minimum final value (Defaults to 0.0).
 * @param epsilon Safeguard for constant (or near constant) time series as the operation implies a unit scale operation
 * between min and max values in the tss.
 */
GAUSSAPI void maxMinNormInPlace(af::array &tss, double high = 1.0, double low = 0.0, double epsilon = 0.00000001);

/**
 * @brief Normalizes the given time series according to its maximum-minimum value and its mean. It follows the following
 * formulae:
 *
 * \f[
 * \acute{x} = \frac{x - mean(x)}{max(x) - min(x)}.
 * \f]
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 *
 * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0) have been
 * normalized by substracting the mean from each number and dividing each number by \f$ max(x) - min(x)\f$, in the
 * time series.
 */
GAUSSAPI af::array meanNorm(const af::array &tss);

/**
 * @brief Normalizes the given time series according to its maximum-minimum value and its mean. It follows the following
 * formulae:
 *
 * \f[
 * \acute{x} = \frac{x - mean(x)}{max(x) - min(x)}.
 * \f]
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 */
GAUSSAPI void meanNormInPlace(af::array &tss);

/**
 * @brief Calculates a new set of timeseries with zero mean and standard deviation one.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param axis Axis of the operation.
 * @param ddof Degrees of freedom for stdev
 *
 * @return af::array With the same dimensions as tss where the time series have been adjusted for zero mean and one as
 * standard deviation.
 */
GAUSSAPI af::array znorm(const af::array &tss, const int axis = 0, const int ddof = 0);

/**
 * @brief Adjusts the time series in the given input and performs z-norm inplace (without allocating further memory).
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param ddof Degrees of freedom for stdev
 */
GAUSSAPI void znormInPlace(af::array &tss, const int ddof = 0);


GAUSSAPI af::array unitLengthNorm(const af::array &tss);
GAUSSAPI af::array medianNorm(const af::array &tss);
GAUSSAPI af::array sigmoidNorm(const af::array &tss);
GAUSSAPI af::array tanhNorm(const af::array &tss);

}  // namespace gauss

#endif