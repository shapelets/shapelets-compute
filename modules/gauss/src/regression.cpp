/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/internal/scopedHostPtr.h>
#include <gauss/regression.h>
#include <gauss/statistics.h>

#include <boost/math/distributions/students_t.hpp>

constexpr auto EPSILON = 1e-20;

void gauss::regression::linear(const af::array &xss, const af::array &yss, af::array &slope, af::array &intercept,
                               af::array &rvalue, af::array &pvalue, af::array &stderrest) {
    auto n = xss.dims(0);

    af::array meanX = af::mean(xss, 0);
    af::array meanY = af::mean(yss, 0);

    af::array sumSquares = af::array(2, 2, xss.dims(1), xss.type());

    // Assuming xss and yss contain the same number of time series
    for (int i = 0; i < xss.dims(1); i++) {
        sumSquares(af::span, af::span, i) =
            gauss::statistics::covariance(af::join(1, xss(af::span, i), yss(af::span, i)));
    }

    af::array ssxm = sumSquares(0, 0, af::span);
    ssxm = af::reorder(ssxm, 0, 2, 1, 3);
    af::array ssxym = sumSquares(0, 1, af::span);
    ssxym = af::reorder(ssxym, 0, 2, 1, 3);
    af::array ssyxm = sumSquares(1, 0, af::span);
    ssyxm = af::reorder(ssyxm, 0, 2, 1, 3);
    af::array ssym = sumSquares(1, 1, af::span);
    ssym = af::reorder(ssym, 0, 2, 1, 3);

    af::array rNum = ssxym;

    af::array rDen = af::sqrt(ssxm * ssym);

    af::array r = af::transpose(af::constant(0, xss.dims(1), xss.type()));
    r = (rDen > 0.0) * rNum / rDen;
    r = af::min(r, 1.0);
    r = af::max(r, -1.0);
    rvalue = r;

    auto df = n - 2;
    slope = rNum / ssxm;
    intercept = meanY - slope * meanX;

    boost::math::students_t dist(df);

    af::array t = r * af::sqrt(df / ((1.0 - r + EPSILON) * (1.0 + r + EPSILON)));
    // Using boost to compute the CDF of the T-Student distribution
    // It would be better to move this computation to the GPU
    // Converting to af::dtype::f32 and back to the original type later on
    // to avoid templating this function and all the ones using it
    auto aux = gauss::utils::makeScopedHostPtr(af::abs(t).as(af::dtype::f32).host<float>());
    for (long i = 0; i < t.dims(1); i++) {
        aux[i] = 2.0f * (1.0f - static_cast<float>(boost::math::cdf(dist, aux[i])));
    }
    pvalue = af::array(1, t.dims(1), aux.get()).as(xss.type());
    stderrest = af::sqrt((1 - af::pow(r, 2)) * ssym / ssxm / df);
}
