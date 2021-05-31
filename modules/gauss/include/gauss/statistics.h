/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_STATISTICS_H
#define GAUSS_STATISTICS_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <optional>
#include <tuple>

namespace gauss::statistics {

enum class XCorrScale { BIASED, UNBIASED, COEFF, NONE };

af::array stdev(const af::array &tss, const unsigned int ddof = 1, const unsigned int dim = 0);
af::array var(const af::array &tss, const unsigned int ddof = 1, const unsigned int dim = 0);
af::array moment(const af::array &tss, unsigned int k, const unsigned int dim = 0);
af::array skewness(const af::array &tss, const unsigned int dim = 0);
af::array kurtosis(const af::array &tss, const unsigned int dim = 0);


af::array covariance(const af::array &x, const unsigned int ddof = 1);
af::array corrcoef(const af::array &x, const unsigned int ddof = 1);


std::tuple<af::array, af::array> xcorr(const af::array &x, 
                                       const af::array &y, 
                                       const std::optional<unsigned int> &maxlag = std::nullopt, 
                                       const std::optional<XCorrScale> &scale = std::nullopt); 

std::tuple<af::array, af::array> xcov(const af::array &x, const af::array &y, 
                                      const std::optional<unsigned int> &maxlag = std::nullopt,
                                      const std::optional<XCorrScale> &scale = std::nullopt);

af::array autocorr(const af::array &x,
                   const std::optional<unsigned int> &maxlag = std::nullopt, 
                   const std::optional<XCorrScale> &scale = std::nullopt);

af::array autocov(const af::array &x,
                  const std::optional<unsigned int> &maxlag = std::nullopt, 
                  const std::optional<XCorrScale> &scale = std::nullopt);

/**
 * @brief The Ljung–Box test checks that data within the time series are independently distributed (i.e. the
 * correlations in the population from which the sample is taken are 0, so that any observed correlations in the data
 * result from randomness of the sampling process). Data are no independently distributed, if they exhibit serial
 * correlation.
 *
 * The test statistic is:
 *
 * \f[
 * Q = n\left(n+2\right)\sum_{k=1}^h\frac{\hat{\rho}^2_k}{n-k}
 * \f]
 *
 * where ''n'' is the sample size, \f$\hat{\rho}k \f$ is the sample autocorrelation at lag ''k'', and ''h'' is the
 * number of lags being tested. Under \f$ H_0 \f$ the statistic Q follows a \f$\chi^2{(h)} \f$. For significance level
 * \f$\alpha\f$, the \f$critical region\f$ for rejection of the hypothesis of randomness is:
 *
 * \f[
 * Q > \chi_{1-\alpha,h}^2
 * \f]
 *
 * where \f$ \chi_{1-\alpha,h}^2 \f$ is the \f$\alpha\f$-quantile of the chi-squared distribution with ''h'' degrees of
 * freedom.
 *
 * [1] G. M. Ljung  G. E. P. Box (1978). On a measure of lack of fit in time series models.
 * Biometrika, Volume 65, Issue 2, 1 August 1978, Pages 297–303.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
 * one indicates the number of time series.
 * @param lags Number of lags being tested.
 *
 * @return af::array Ljung-Box statistic test.
 */
GAUSSAPI af::array ljungBox(const af::array &tss, const std::optional<unsigned int> &maxlag = std::nullopt);


// /**
//  * @brief Returns the covariance matrix of the time series contained in tss.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param unbiased Determines whether it divides by n - 1 (if false) or n (if true).
//  *
//  * @return af::array The covariance matrix of the time series.
//  */
// GAUSSAPI af::array covariance(const af::array &tss, bool unbiased = true);

// /**
//  * @brief Returns the kth moment of the given time series.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param k The specific moment to be calculated.
//  *
//  * @return af::array The kth moment of the given time series.
//  */
// GAUSSAPI af::array moment(const af::array &tss, unsigned int k);

// /**
//  * @brief Returns the kurtosis of tss (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  *
//  * @return af::array The kurtosis of tss.
//  */
// GAUSSAPI af::array kurtosis(const af::array &tss);

// /**
//  * @brief Calculates the sample skewness of tss (calculated with the adjusted Fisher-Pearson standardized moment
//  * coefficient G1).
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  *
//  * @return af::array Array containing the skewness of each time series in tss.
//  */
// GAUSSAPI af::array skewness(const af::array &tss);



// using AggregationFuncDimT = af::array (*)(const af::array &, const dim_t);
// using AggregationFuncBoolDimT = af::array (*)(const af::array &, bool, const dim_t);
// using AggregationFuncInt = af::array (*)(const af::array &, const int);

// /**
//  * @brief Calculates the value of an aggregation function f_agg (e.g. var or mean) of the autocorrelation
//  * (Compare to http://en.wikipedia.org/wiki/Autocorrelation#Estimation), taken over different all possible
//  * lags (1 to length of x).
//  * \f[
//  * \frac{1}{n-1} \sum_{l=1,\ldots, n} \frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu),
//  * \f]
//  *  where \f$n\f$ is the length of the time series \f$X_i\f$, \f$\sigma^2\f$ its variance and \f$\mu\f$ its mean.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
//  * dimension one indicates the number of time series.
//  * @param aggregationFunction The function to summarise all autocorrelation with different lags.
//  *
//  * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0)
//  * contains the aggregated correlation for each time series.
//  */
// GAUSSAPI af::array aggregatedAutocorrelation(const af::array &tss, AggregationFuncBoolDimT aggregationFunction);

// /**
//  * @brief Calculates the value of an aggregation function f_agg (e.g. var or mean) of the autocorrelation
//  * (Compare to http://en.wikipedia.org/wiki/Autocorrelation#Estimation), taken over different all possible
//  * lags (1 to length of x).
//  * \f[
//  * \frac{1}{n-1} \sum_{l=1,\ldots, n} \frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu),
//  * \f]
//  *  where \f$n\f$ is the length of the time series \f$X_i\f$, \f$\sigma^2\f$ its variance and \f$\mu\f$ its mean.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
//  * dimension one indicates the number of time series.
//  * @param aggregationFunction The function to summarise all autocorrelation with different lags.
//  *
//  * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0)
//  * contains the aggregated correlation for each time series.
//  */
// GAUSSAPI af::array aggregatedAutocorrelation(const af::array &tss, AggregationFuncDimT aggregationFunction);

// /**
//  * @brief Calculates the value of an aggregation function f_agg (e.g. var or mean) of the autocorrelation
//  * (Compare to http://en.wikipedia.org/wiki/Autocorrelation#Estimation), taken over different all possible
//  * lags (1 to length of x).
//  * \f[
//  * \frac{1}{n-1} \sum_{l=1,\ldots, n} \frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu),
//  * \f]
//  *  where \f$n\f$ is the length of the time series \f$X_i\f$, \f$\sigma^2\f$ its variance and \f$\mu\f$ its mean.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
//  * dimension one indicates the number of time series.
//  * @param aggregationFunction The function to summarise all autocorrelation with different lags.
//  *
//  * @return af::array An array with the same dimensions as tss, whose values (time series in dimension 0)
//  * contains the aggregated correlation for each time series.
//  */
// GAUSSAPI af::array aggregatedAutocorrelation(const af::array &tss, AggregationFuncInt aggregationFunction);

// /**
//  * @brief Calculates the value of the partial autocorrelation function at the given lag. The lag \f$k\f$ partial
//  * autocorrelation of a time series \f$\lbrace x_t, t = 1 \ldots T \rbrace\f$ equals the partial correlation of
//  * \f$x_t\f$ and \f$x_{t-k}\f$, adjusted for the intermediate variables \f$\lbrace x_{t-1}, \ldots, x_{t-k+1}
//  * \rbrace\f$ ([1]). Following [2], it can be defined as:
//  * \f[
//  *      \alpha_k = \frac{ Cov(x_t, x_{t-k} | x_{t-1}, \ldots, x_{t-k+1})}
//  *      {\sqrt{ Var(x_t | x_{t-1}, \ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \ldots, x_{t-k+1} )}}
//  * \f]
//  * with (a) \f$x_t = f(x_{t-1}, \ldots, x_{t-k+1})\f$ and (b) \f$ x_{t-k} = f(x_{t-1}, \ldots, x_{t-k+1})\f$
//  * being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
//  * predict \f$ x_t \f$ whereas in (b), future values are used to calculate the past value \f$x_{t-k}\f$.
//  * It is said in [1] that, for an AR(p), the partial autocorrelations \f$ \alpha_k \f$ will be nonzero for
//  * \f$ k<=p \f$ and zero for \f$ k>p \f$. With this property, it is used to determine the lag of an AR-Process.
//  *
//  * [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control.
//  * John Wiley & Sons.
//  *
//  * [2] https://onlinecourses.science.psu.edu/stat510/node/62
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param lags Indicates the lags to be calculated.
//  *
//  * @return af::array The partial autocorrelation for each time series for the given lag.
//  */
// GAUSSAPI af::array partialAutocorrelation(const af::array &tss, const af::array &lags);

// /**
//  * @brief Calculates the autocorrelation of the specified lag for the given time series, according to the formula [1].
//  * \f[
//  * \frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu),
//  * \f]
//  * where \f$n\f$ is the length of the time series \f$X_i\f$, \f$\sigma^2\f$ its variance and \f$\mu\f$ its mean, \f$l\f$
//  * denotes the lag.
//  *
//  * [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param maxLag The maximum lag to compute.
//  * @param unbiased Determines whether it divides by (n - lag) (if true), or n (if false).
//  *
//  * @return af::array The autocorrelation value for the given time series.
//  */
// GAUSSAPI af::array autoCorrelation(const af::array &tss, unsigned int maxLag, bool unbiased = false);

// /**
//  * @brief Calculates the cross-correlation of the given time series.
//  *
//  * @param xss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param yss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param unbiased Determines whether it divides by n - lag (if true) or n (if false).
//  *
//  * @return af::array The cross-correlation value for the given time series.
//  */
// GAUSSAPI af::array crossCorrelation(const af::array &xss, const af::array &yss, bool unbiased = false);

// /**
//  * @brief Calculates the auto-covariance the given time series.
//  *
//  * @param xss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param unbiased Determines whether it divides by n - lag (if true) or n (if false).
//  *
//  * @return af::array The auto-covariance value for the given time series.
//  */
// GAUSSAPI af::array autoCovariance(const af::array &xss, bool unbiased = false);


// /**
//  * @brief Calculates the cross-covariance of the given time series.
//  *
//  * @param xss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param yss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series.
//  * @param unbiased Determines whether it divides by n - lag (if true) or n (if false).
//  *
//  * @return af::array The cross-covariance value for the given time series.
//  */
// GAUSSAPI af::array crossCovariance(const af::array &xss, const af::array &yss, bool unbiased = false);


// /**
//  * @brief Computes the correlation coeficients of all the column vectors in tss
//  */ 
// GAUSSAPI af::array correlation(const af::array &tss, bool unbiased = false);



// /**
//  * @brief Returns values at the given quantile.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series. NOTE: the time series should be sorted.
//  * @param q Percentile(s) at which to extract score(s). One or many.
//  * @param precision Number of decimals expected.
//  *
//  * @return af::array Values at the given quantile.
//  */
// GAUSSAPI af::array quantile(const af::array &tss, const af::array &q, float precision = 100000000);

// /**
//  * @brief Discretizes the time series into equal-sized buckets based on sample quantiles.
//  *
//  * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and dimension
//  * one indicates the number of time series. NOTE: the time series should be sorted.
//  * @param quantiles Number of quantiles to extract. From 0 to 1, step 1/quantiles.
//  * @param precision Number of decimals expected.
//  *
//  * @return af::array Matrix with the categories, one category per row, the start of the category in the first column and
//  * the end in the second category.
//  */
// GAUSSAPI af::array quantilesCut(const af::array &tss, float quantiles, float precision = 0.00000001);

}  // namespace gauss

#endif
