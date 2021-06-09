/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_REGRESSION_H
#define GAUSS_REGRESSION_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::regression {

/**
 * @brief Calculate a linear least-squares regression for two sets of measurements. Both arrays should have the same
 * length.
 *
 * @param xss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param yss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param slope Slope of the regression line.
 * @param intercept Intercept of the regression line.
 * @param rvalue Correlation coefficient.
 * @param pvalue Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald
 * Test with t-distribution of the test statistic.
 * @param stderrest Standard error of the estimated gradient.
 */
GAUSSAPI void linear(const af::array &xss, const af::array &yss, af::array &slope, af::array &intercept,
                     af::array &rvalue, af::array &pvalue, af::array &stderrest);

}  // namespace gauss

#endif
