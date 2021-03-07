#ifndef GAUSS_RANDOM_H
#define GAUSS_RANDOM_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <optional>

namespace gauss::random {

    /**
     * Return random integers from low (inclusive) to high (exclusive).
     *
     * Return random integers from the “discrete uniform” distribution of the specified
     * dtype in the “half-open” interval [low, high). If high is undefined,
     * then results are from [0, low).
     *
     * @param low       Lowest (signed) integers to be drawn from the distribution
     *                  (unless high is undefined, in which case this parameter is one
     *                  above the highest such integer).
     * @param high      If provided, one above the largest (signed) integer to be drawn from the
     *                  distribution (see above for behavior if high is undefined).
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array randint(int64_t low,
                               const std::optional<int64_t> &high,
                               const af::dim4 &shape,
                               const af::dtype &dtype = af::dtype::s32,
                               const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     * Draw samples from a logistic distribution.
     *
     * Samples are drawn from a logistic distribution with specified parameters,
     * loc (location or mean, also median), and scale (>0).
     *
     * @param loc       Parameter of the distribution.
     * @param scale     Parameter of the distribution. Must be non-negative.
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array logistic(double loc,
                                double scale,
                                const af::dim4 &shape,
                                const af::dtype &dtype = af::dtype::f32,
                                const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     *
     * Draw samples from a log-normal distribution.
     *
     * Draw samples from a log-normal distribution with specified mean, standard deviation, and array shape.
     * Note that the mean and standard deviation are not the values for the distribution itself,
     * but of the underlying normal distribution it is derived from.
     *
     * @param mean      Mean value of the underlying normal distribution.
     * @param sigma     Standard deviation of the underlying normal distribution. Must be non-negative
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array lognormal(double mean,
                                 double sigma,
                                 const af::dim4 &shape,
                                 const af::dtype &dtype = af::dtype::f32,
                                 const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     *
     * Draw random samples from a normal (Gaussian) distribution.
     *
     * Draw random samples from a normal distribution. with specified mean, standard deviation, and array shape.
     *
     * @param mean      Mean value of the normal distribution.
     * @param sigma     Standard deviation of the normal distribution. Must be non-negative
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array normal(double mean,
                              double sigma,
                              const af::dim4 &shape,
                              const af::dtype &dtype = af::dtype::f32,
                              const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     * Draw samples from a Wald, or inverse Gaussian, distribution.
     *
     * As the scale approaches infinity, the distribution becomes more like a Gaussian.
     *
     * @param mean      Distribution mean, must be > 0
     * @param scale     Scale parameter, must be > 0.
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array wald(double mean,
                            double scale,
                            const af::dim4 &shape,
                            const af::dtype &dtype = af::dtype::f32,
                            const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     * Draw samples from a gamma distribution, using Marsaglia and Tsang method.
     *
     * @param alpha     It is normally called shape (but we use it here with a different meaning) or
     *                  also known as the `K` parameter of the gamma distribution.
     * @param lambda    It is the inverse of the scale parameter (1/scale).
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array gamma(double alpha,
                             double lambda,
                             const af::dim4 &shape,
                             const af::dtype &dtype = af::dtype::f32,
                             const std::optional<af::randomEngine> &engine = std::nullopt);

    /**
     * Draw samples from a uniform distribution over the half-open interval [low, high)
     *
     * @param low       Inclusive low value
     * @param high      Exclusive high value
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array uniform(double low,
                               double high,
                               const af::dim4 &shape,
                               const af::dtype &dtype = af::dtype::f32,
                               const std::optional<af::randomEngine> &engine = std::nullopt);

    /**
     * Draw samples from an exponential distribution.
     *
     * @param scale     Scale is the inverse of lambda.
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array exponential(double scale,
                                   const af::dim4 &shape,
                                   const af::dtype &dtype = af::dtype::f32,
                                   const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     * Draw samples from an chi-square distribution.
     *
     * @param df        Degrees of freedom.
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array chisquare(double df,
                                 const af::dim4 &shape,
                                 const af::dtype &dtype = af::dtype::f32,
                                 const std::optional<af::randomEngine> &engine = std::nullopt);

    /**
     * Draw samples from a Beta distribution.
     *
     * The Beta distribution is a special case of the Dirichlet distribution,
     * and is related to the Gamma distribution
     *
     * @param alpha     Alpha, positive
     * @param beta      Beta, positive
     * @param shape     Output shape. If the given shape is, e.g., (m, n, k),
     *                  then m * n * k samples are drawn.
     * @param dtype     Desired dtype of the result.  Defaults to f32.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array of the given `dtype` and `shape`.
     */
    GAUSSAPI af::array beta(double alpha,
                            double beta,
                            const af::dim4 &shape,
                            const af::dtype &dtype = af::dtype::f32,
                            const std::optional<af::randomEngine> &engine = std::nullopt);


    /**
     * Draw random samples from a multivariate normal distribution.
     *
     * The multivariate normal, multinormal or Gaussian distribution is a generalization of
     * the one-dimensional normal distribution to higher dimensions. Such a distribution is
     * specified by its mean and covariance matrix. These parameters are analogous to the
     * mean (average or “center”) and variance (standard deviation, or “width,” squared)
     * of the one-dimensional normal distribution.
     *
     * This method is based on a Cholesky decomposition of the covariance matrix; to increase
     * the numerical stability of the algorithm, a small eps factor will be added.
     *
     *
     * @param samples   Number of samples to draw
     * @param mean      For each series, it indicates the mean.
     * @param cov       A covariance matrix.
     * @param engine    Configured engine for seeding the generation.  Defaults to empty option
     * @return          An array with as many rows as samples and with as many columns as means
     */
    GAUSSAPI af::array multivariate_normal(int64_t samples,
                                           const af::array& mean,
                                           const af::array& cov,
                                           const std::optional<af::randomEngine> &engine = std::nullopt);
}


#endif   // GAUSS_RANDOM_H
