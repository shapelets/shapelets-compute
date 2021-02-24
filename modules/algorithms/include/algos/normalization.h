#ifndef ALGOS_CORE_NORMALIZATION_H
#define ALGOS_CORE_NORMALIZATION_H

#include <arrayfire.h>
#include <algos/defines.h>

namespace algos {

namespace normalization {

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
ALGOSAPI af::array decimalScalingNorm(const af::array &tss);

/**
 * @brief Same as decimalScalingNorm, but it performs the operation in place, without allocating further memory.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 */
ALGOSAPI void decimalScalingNormInPlace(af::array &tss);

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
ALGOSAPI af::array maxMinNorm(const af::array &tss, double high = 1.0, double low = 0.0, double epsilon = 0.00000001);

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
ALGOSAPI void maxMinNormInPlace(af::array &tss, double high = 1.0, double low = 0.0, double epsilon = 0.00000001);

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
ALGOSAPI af::array meanNorm(const af::array &tss);

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
ALGOSAPI void meanNormInPlace(af::array &tss);

/**
 * @brief Calculates a new set of timeseries with zero mean and standard deviation one.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param epsilon Minimum standard deviation to consider. It acts as a gatekeeper for those time series that may be
 * constant or near constant.
 *
 * @return af::array With the same dimensions as tss where the time series have been adjusted for zero mean and one as
 * standard deviation.
 */
ALGOSAPI af::array znorm(const af::array &tss, double epsilon = 0.00000001);

/**
 * @brief Adjusts the time series in the given input and performs z-norm inplace (without allocating further memory).
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param epsilon Minimum standard deviation to consider. It acts as a gatekeeper for those time series that may be
 * constant or near constant.
 */
ALGOSAPI void znormInPlace(af::array &tss, double epsilon = 0.00000001);

}  // namespace normalization
}  // namespace algos

#endif
