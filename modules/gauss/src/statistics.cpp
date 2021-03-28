#include <gauss/statistics.h>
#include <gauss/linalg.h>

af::array sampleStdev(const af::array &tss) {
    auto n = static_cast<double>(tss.dims(0));
    af::array mean = af::mean(tss, 0);
    return af::sqrt(af::sum(af::pow(tss - af::tile(mean, static_cast<unsigned int>(tss.dims(0))), 2), 0) / (n - 1));
}

af::array gauss::statistics::aggregatedAutocorrelation(const af::array &tss,
                                                     AggregationFuncBoolDimT aggregationFunction) {
    auto n = tss.dims(0);
    af::array autocorrelations = gauss::statistics::autoCorrelation(tss, n, true)(af::seq(1, n - 1), af::span);
    return aggregationFunction(autocorrelations, true, 0);
}

af::array gauss::statistics::aggregatedAutocorrelation(const af::array &tss, AggregationFuncDimT aggregationFunction) {
    auto n = tss.dims(0);
    af::array autocorrelations = gauss::statistics::autoCorrelation(tss, n, true)(af::seq(1, n - 1), af::span);
    return aggregationFunction(autocorrelations, 0);
}

af::array gauss::statistics::aggregatedAutocorrelation(const af::array &tss, AggregationFuncInt aggregationFunction) {
    auto n = tss.dims(0);
    af::array autocorrelations = gauss::statistics::autoCorrelation(tss, n, true)(af::seq(1, n - 1), af::span);
    return aggregationFunction(autocorrelations, 0);
}

af::array gauss::statistics::ljungBox(const af::array &tss, unsigned int lags) {
    auto n = tss.dims(0);
    const double e = 2;
    af::array ac = gauss::statistics::autoCorrelation(tss, lags + 1);
    af::array acp = af::pow(ac(af::seq(1, af::end), af::span), e);
    af::array r = af::range(acp.dims(0), acp.dims(1), acp.dims(2), acp.dims(3)) + 1;
    af::array d = acp / (n - r);
    return af::sum(d) * n * (n + 2);
}

af::array gauss::statistics::quantile(const af::array &tss, const af::array &q, float precision) {
    auto n = tss.dims(0);

    af::array idx = q * (n - 1);
    af::array idxAsInt = idx.as(af::dtype::u32);

    af::array a = af::tile(idxAsInt == idx, 1, static_cast<unsigned int>(tss.dims(1))) * tss(idx, af::span);
    af::array fraction = (idx * precision) % precision / precision;

    af::array b = af::tile(idxAsInt != idx, 1, static_cast<unsigned int>(tss.dims(1))) * tss(idxAsInt, af::span) +
                  (tss(idxAsInt + 1, af::span) - tss(idxAsInt, af::span)) *
                      af::tile(fraction, 1, static_cast<unsigned int>(tss.dims(1)));
    return a + b;
}

af::array searchSorted(const af::array &tss, const af::array &qs) {
    af::array input = af::tile(tss, 1, 1, static_cast<unsigned int>(qs.dims(0)));
    af::array qsReordered = af::tile(af::reorder(qs, 2, 1, 0, 3), static_cast<unsigned int>(tss.dims(0)));

    af::array result = af::flip(af::sum(input < qsReordered, 2), 0);

    result(0, af::span) += 1;

    return result;
}

af::array gauss::statistics::quantilesCut(const af::array &tss, float quantiles, float precision) {
    af::array q = af::seq(0, 1, 1 / (double)quantiles);
    af::array qs = gauss::statistics::quantile(tss, q);
    af::array ss = searchSorted(tss, qs);

    af::array qcut = af::array(qs.dims(0) - 1, 2, qs.dims(1), qs.type());
    qcut(af::span, 0, af::span) = af::reorder(qs(af::seq(0, static_cast<double>(qs.dims(0)) - 2), af::span), 0, 2, 1, 3);
    qcut(af::span, 1, af::span) = af::reorder(qs(af::seq(1, static_cast<double>(qs.dims(0)) - 1), af::span), 0, 2, 1, 3);

    qcut(0, 0, af::span) -= precision;

    af::array result = af::array(tss.dims(0), 2, tss.dims(1), tss.type());

    // With a parallel GFOR we cannot index by the matrix ss. It flattens it by default
    // gfor(af::seq i, tss.dims(1)) {
    for (int i = 0; i < tss.dims(1); i++) {
        result(af::span, af::span, i) = qcut(ss(af::span, i) - 1, af::span, i);
    }

    return result;
}

af::array gauss::statistics::skewness(const af::array &tss) {
    auto n = static_cast<double>(tss.dims(0));
    af::array tssMinusMean = (tss - af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0))));
    af::array m3 = gauss::statistics::moment(tssMinusMean, 3);
    af::array s3 = af::pow(sampleStdev(tss), 3);
    return (std::pow(n, 2.0) / ((n - 1) * (n - 2))) * m3 / s3;
}

af::array gauss::statistics::kurtosis(const af::array &tss) {
    auto n = static_cast<double>(tss.dims(0));

    double a = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
    af::array tssMinusMean = (tss - af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0))));
    af::array sstdev = af::tile(sampleStdev(tss), static_cast<unsigned int>(tss.dims(0)));
    af::array b = af::sum(af::pow(tssMinusMean / sstdev, 4), 0);
    double c = (3 * std::pow(n - 1, 2)) / ((n - 2) * (n - 3));

    return a * b - c;
}

af::array gauss::statistics::moment(const af::array &tss, unsigned int k) {
    auto n = static_cast<double>(tss.dims(0));
    return af::sum(af::pow(tss, static_cast<double>(k)), 0) / n;
}

af::array gauss::statistics::correlation(const af::array &tss, bool unbiased) {
    auto stds = af::stdev(tss, 0);
    auto divisor = af::matmulTN(stds, stds);
    auto covs = covariance(tss, unbiased);
    return covs / divisor;
}

af::array gauss::statistics::covariance(const af::array &tss, bool unbiased) {
    auto n = tss.dims(0);
    af::array result = gauss::statistics::crossCovariance(tss, tss) * n / (n - unbiased);
    return af::reorder(result(0, af::span, af::span), 1, 2, 0, 3);
}

af::array gauss::statistics::crossCovariance(const af::array &xss, const af::array &yss, bool unbiased) {
    // To be used as divisor if unbiased is false
    long n = static_cast<long>(xss.dims(0));
    // To be used as divisor if unbiased is true and also to determine the size of the output
    long nobs = static_cast<long>(std::max(xss.dims(0), yss.dims(0)));

    // Mean value of each time series contained in xss
    af::array meanXss = af::mean(xss, 0);
    // Mean value of each time series contained in yss
    af::array meanYss = af::mean(yss, 0);

    // Substracting the mean to all the elements in xss for all the time series
    af::array xsso = xss - af::tile(meanXss, static_cast<unsigned int>(xss.dims(0)));
    // Substracting the mean to all the elements in yss flipped for all the time series.
    // The flip operation is required because we are using convolve later on
    af::array ysso = af::flip(yss, 0) - af::tile(meanYss, static_cast<unsigned int>(yss.dims(0)));

    af::array d;

    // Determining which divisor to use
    if (unbiased) {
        d = af::flip(af::tile((af::range(nobs) + 1.0).as(xss.type()), 1, static_cast<unsigned int>(xss.dims(1))), 0);
    } else {
        d = af::constant(n, nobs, xss.dims(1), xss.type());
    }

    // The result is a cube with nobs in the first dimensions, that determines the number of lags.
    // And the number of time series in yss as 2nd dimension and the number of time series in
    //  xss as the 3rd dimension
    af::array result = af::array(nobs, yss.dims(1), xss.dims(1), xss.type());
    gfor(af::seq i, static_cast<double>(xss.dims(1))) {
        // Flipping the result of the convolve operation because we flipped the input data
        result(af::span, af::span, i, af::span) =
            af::flip(af::convolve(xsso(af::span, i), ysso, AF_CONV_EXPAND)(af::seq(nobs), af::span), 0) / d;
    }

    return result;
}

af::array gauss::statistics::crossCorrelation(const af::array &xss, const af::array &yss, bool unbiased) {
    // Standard deviation of the time series in xss
    af::array stdevXss = af::stdev(xss, 0);
    // Standard deviation of the time series in yss
    af::array stdevYss = af::stdev(yss, 0);

    // Cross covariance of the time series contained in css and yss
    af::array ccov = gauss::statistics::crossCovariance(xss, yss, unbiased);

    // Dviding by the product of their standard deviations
    return ccov / af::tile(stdevXss * stdevYss, static_cast<unsigned int>(ccov.dims(0)), 1,
                           static_cast<unsigned int>(ccov.dims(1)));
}


af::array gauss::statistics::autoCorrelation(const af::array &tss, unsigned int maxLag, bool unbiased) {
    // Calculating the auto covariance of tss
    af::array acov = gauss::statistics::autoCovariance(tss, unbiased);

    // Slicing up to maxLag and normalizing by the value of lag 0
    return acov(af::seq(maxLag), af::span) / af::tile(acov(0, af::span), maxLag);
}

af::array gauss::statistics::autoCovariance(const af::array &xss, bool unbiased) {
    // Matrix with number of points in xss as the 1st dimension and the number of time
    // series in yss as the 2nd dimension
    af::array result = af::array(xss.dims(0), xss.dims(1), xss.type());
    // Calculating all the covariances in parallel, returning only the first slice of
    // the cube since the others are just calculations that are not required.
    // With a sequential for loop we would remove such calculations, but it might be slower
    result = gauss::statistics::crossCovariance(xss, xss, unbiased)(af::span, af::span, 0);
    return result;
}

af::array gauss::statistics::partialAutocorrelation(const af::array &tss, const af::array &lags) {
    auto n = tss.dims(0);
    af::array m = af::max(lags, 0);
    dim_t maxlag = m.scalar<int>();

    af::array ld;
    if (n < 1) {
        ld = af::constant(af::NaN, maxlag + 1, tss.dims(1), tss.type());
    } else {
        if (n <= maxlag) {
            maxlag = n - 1;
        }
        af::array acv = gauss::statistics::autoCovariance(tss, true);
        ld = gauss::linalg::levinsonDurbin(acv, maxlag);
    }
    return ld;
}


