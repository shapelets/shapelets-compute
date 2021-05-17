#include <gauss/internal/scopedHostPtr.h>
#include <gauss/random.h>
#include <iostream>

#define FLOAT32_EPS 1.1920929e-07

af::array gauss::random::randint(int64_t low,
                                 const std::optional<int64_t>& high,
                                 const af::dim4 &shape,
                                 const af::dtype &dtype,
                                 const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());

    auto l = 0.;
    auto h = static_cast<double>(low);

    if (high.has_value()) {
        l = static_cast<double>(low);
        h = static_cast<double>(high.value());
    }

    auto u = af::randu(shape, af::dtype::f32, re);

    // The data is uniformly distributed between [0, 1], hence the adjustment before
    // translating the result to h && l
    u = (af::min(u, 1.0 - FLOAT32_EPS) * (h - l)) + l;

    return u.as(dtype);
}

af::array gauss::random::logistic(double loc, double scale,
                                  const af::dim4 &shape,
                                  const af::dtype &dtype,
                                  const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto u = af::randu(shape, af::dtype::f32, re);

    // Intervals should be open [0, 1)
    u = af::min(u, 1.0 - FLOAT32_EPS);

    return loc - (scale * af::log(1.0 / u - 1.0));
}

af::array gauss::random::lognormal(double mean,
                                   double sigma,
                                   const af::dim4 &shape,
                                   const af::dtype &dtype,
                                   const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto u = af::randn(shape, af::dtype::f32, re);
    return af::exp(mean + (sigma * u));
}

af::array gauss::random::normal(double mean,
                                double sigma,
                                const af::dim4 &shape,
                                const af::dtype &dtype,
                                const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto u = af::randn(shape, af::dtype::f32, re);
    return mean + (sigma * u);
}

af::array gauss::random::wald(double mean,
                              double scale,
                              const af::dim4 &shape,
                              const af::dtype &dtype,
                              const std::optional<af::randomEngine> &engine) {

    auto n = shape.elements();
    auto re = engine.value_or(af::getDefaultRandomEngine());

    auto y = af::pow(af::randn(shape, af::dtype::f32, re), 2);
    auto u = af::randu(af::dim4(n), af::dtype::f32, re);

    auto mu2 = std::pow(mean, 2);
    auto x = mean + mu2 / (2.0 * scale) * y - mean / (2.0 * scale) * sqrt(4.0 * mean * scale * y + mu2 * y * y);
    auto reject_index = u > (mean / (mean + x));
    x(reject_index) = mu2 / x(reject_index);

    if (x.type() != dtype)
        x = x.as(dtype);

    return af::moddims(x, shape);
}

af::array gauss::random::gamma(double alpha,
                               double lambda,
                               const af::dim4 &shape,
                               const af::dtype &dtype,
                               const std::optional<af::randomEngine> &engine) {

    auto n = shape.elements();
    auto re = engine.value_or(af::getDefaultRandomEngine());

    if (alpha < 1.0) {
        auto r = gamma(alpha + 1.0, lambda, shape, dtype, engine);
        return r * af::pow(af::randu(af::dim4(n), af::dtype::f32, re), 1.0 / alpha);
    }

    // this is where the result is to be calculated
    auto result = af::constant(0.0, n, af::dtype::f32);

//
//      implementation is based on matlab code
//
//      function x=gamrand(alpha,lambda)
//      % Gamma(alpha,lambda) generator using Marsaglia and Tsang method
//      % Algorithm 4.33
//      if alpha>1
//          d=alpha-1/3; c=1/sqrt(9*d); flag=1;
//          while flag
//              Z=randn;
//              if Z>-1/c
//                  V=(1+c*Z)^3; U=rand;
//                  flag=log(U)>(0.5*Z^2+d-d*V+d*log(V));
//              end
//          end
//          x=d*V/lambda;
//      else
//          x=gamrand(alpha+1,lambda);
//          x=x*rand^(1/alpha);
//      end
//

    auto d = alpha - 1. / 3.;
    auto c = 1.0 / sqrt(9.0 * d);
    auto ii = 0L;

    while (ii < n) {
        auto left = n - ii;
        auto z = af::randn(af::dim4(left), af::dtype::f32, re);
        auto y = (1.0 + c * z);
        auto v = y * y * y;
        auto if_test = ((z >= -1.0 / c) && (v > 0.0));
        auto Z = z(if_test).copy();
        auto V = v(if_test).copy();
        auto U = af::randu(af::dim4(V.elements()), af::dtype::f32, re);
        auto flag = U < af::exp((0.5 * Z * Z + d - d * V + d * af::log(V)));
        auto x = d * V(flag) / lambda;
        auto accepted = x.elements();
        auto left_index = af::seq(ii, std::min(n, ii + accepted)-1);
        auto right_index = af::seq(0, std::min(left, accepted)-1);
        result(left_index) = x(right_index);
        ii += accepted;
    }

    if (result.type() != dtype)
        result = result.as(dtype);

    return af::moddims(result, shape);
}

af::array gauss::random::permute(const af::array& original,
                                 const dim_t axis,
                                 const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto len = original.dims(axis);
    auto tmp = af::randu(af::dim4(len, 1, 1, 1), af::dtype::f32, re);
    af::array val, idx;
    af::sort(val, idx, tmp);
    switch (axis) {
        case 0:
            return original(idx, af::span, af::span, af::span);
        case 1:
            return original(af::span, idx, af::span, af::span);
        case 2:
            return original(af::span, af::span, idx, af::span);
        default:
            return original(af::span, af::span, af::span, idx);
    }
}

af::array gauss::random::uniform(double low,
                                 double high,
                                 const af::dim4 &shape,
                                 const af::dtype &dtype,
                                 const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    af::array v = af::randu(shape, dtype, re);
    return (v * (high - low)) + low;
}


af::array gauss::random::exponential(double scale,
                                     const af::dim4 &shape,
                                     const af::dtype &dtype,
                                     const std::optional<af::randomEngine> &engine) {

    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto v = af::randu(shape, dtype, re);
    v = af::min(v, 1.0 - FLOAT32_EPS);
    return af::log(1.0 - v) * (-scale);
}


af::array gauss::random::chisquare(double df,
                                   const af::dim4 &shape,
                                   const af::dtype &dtype,
                                   const std::optional<af::randomEngine> &engine) {

    return gamma(df / 2.0, 0.5, shape, dtype, engine);
}

af::array gauss::random::beta(double alpha,
                              double beta,
                              const af::dim4 &shape,
                              const af::dtype &dtype,
                              const std::optional<af::randomEngine> &engine) {

    auto x = gamma(alpha, 1.0, shape, dtype, engine);
    auto y = gamma(beta, 1.0, shape, dtype, engine);
    return x / (x + y);
}


af::array gauss::random::multivariate_normal(int64_t samples,
                                             const af::array& mean,
                                             const af::array& cov,
                                             const std::optional<af::randomEngine> &engine) {

    // this is the number of series to generate
    auto d = mean.elements();

    auto dtype = mean.type();

    // Independently of how means is provided (column or row vector), treat it
    // as a row vector (1, d) to ensure there are no misunderstandings on how
    // the algorithm processes data.
    auto check_mean = af::moddims(mean, 1, d);

    if (cov.dims(0) != cov.dims(1))
        throw std::runtime_error("Either the covariance matrix is not rectangular");

    if (cov.dims(0) != d)
        throw std::runtime_error("The dimensions of the mean vector and the covariance matrix doesn't match");

    if (cov.type() != dtype)
        throw std::runtime_error("The types of the mean and covariance matrices do not match.");

    // induce epsilon to alleviate the possibility of introducing numerical problems
    auto cov_e = cov + (af::identity(d, d) * 0.0001);

    af::array cho;
    auto failed_rank = af::cholesky(cho, cov_e);
    if (failed_rank != 0)
        throw std::runtime_error("Cholesky decomposition of the covariance matrix failed at rank " +
            std::to_string(failed_rank));

    // draw normal samples into a matrix of `samples` rows by `d` columns
    auto re = engine.value_or(af::getDefaultRandomEngine());
    auto u = af::randn(af::dim4(samples, d), dtype, re);
    
    af::gforSet(true);
    auto result = check_mean + af::matmulNT(u, cho);
    af::gforSet(false);

    return result;
}
