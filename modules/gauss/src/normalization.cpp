/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/normalization.h>
#include <gauss/statistics.h>

void clip_near_zero(af::array& arr) {
    const auto type = arr.type();
    auto eps = 1e-4;
    if (( type == af::dtype::f64) || (type == af::dtype::c64) || (type == af::dtype::s64) || (type == af::dtype::u64)) 
        eps = 1e-8;
    
    arr(af::abs(arr) < eps) = eps;
}

af::array gauss::normalization::detrend(const af::array &tss) {
    return tss - af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
}

af::array gauss::normalization::decimalScalingNorm(const af::array &tss) {
    auto maxAbs = af::max(af::abs(tss), 0);
    auto const10 = af::constant(10, 1, maxAbs.dims(1));
    auto d = af::ceil(af::log10(maxAbs));
    auto divFactor = af::pow(const10, d);
    return tss / af::tile(divFactor, tss.dims(0));
}

void gauss::normalization::decimalScalingNormInPlace(af::array &tss) {
    auto maxAbs = af::max(af::abs(tss), 0);
    auto const10 = af::constant(10, 1, tss.dims(1));
    auto d = af::ceil(af::log10(maxAbs));
    auto divFactor = af::pow(const10, d);
    tss /= af::tile(divFactor, static_cast<unsigned int>(tss.dims(0)));
}

af::array gauss::normalization::maxMinNorm(const af::array &tss, double high, double low, double _) {
    auto max = af::max(tss, 0);
    auto min = af::min(tss, 0);
    auto diff = max - min;
    clip_near_zero(diff);
    auto min_tiled = af::tile(min, tss.dims(0));
    auto diff_tiled = af::tile(diff, tss.dims(0));
    return low + (((high - low) * (tss - min_tiled)) / diff_tiled);
}

void gauss::normalization::maxMinNormInPlace(af::array &tss, double high, double low, double epsilon) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto diff = max - min;
    clip_near_zero(diff);
    tss -= min;
    tss *= (high - low);
    tss /= diff;
    tss += low;
}

af::array gauss::normalization::meanNorm(const af::array &tss) {
    auto max = af::max(tss, 0);
    auto min = af::min(tss, 0);
    auto diff = max - min;
    clip_near_zero(diff);
    auto mean = af::tile(af::mean(tss, 0), tss.dims(0));
    return (tss - mean )/ af::tile(diff, tss.dims(0));
}

void gauss::normalization::meanNormInPlace(af::array &tss) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto diff = max - min;
    clip_near_zero(diff);
    tss = tss - mean;
    tss = tss / diff;
}

af::array gauss::normalization::znorm(const af::array &tss, const int axis, const int ddof) {
    auto tile_dims = af::dim4(1,1,1,1);
    tile_dims[axis] = tss.dims(axis);

    auto mean = af::mean(tss, axis);
    auto stdev = gauss::statistics::stdev(tss, ddof, axis); 
    clip_near_zero(stdev);
    return (tss - af::tile(mean, tile_dims)) / af::tile(stdev, tile_dims);
}

void gauss::normalization::znormInPlace(af::array &tss, const int ddof) {
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto stdev = gauss::statistics::stdev(tss, ddof, 0);
    clip_near_zero(stdev);
    tss -= mean;
    tss /= stdev;
}

af::array gauss::normalization::unitLengthNorm(const af::array &tss) {
    auto sq = af::pow(tss, 2.0);
    auto sm = af::sum(sq, 0);
    auto norms = af::sqrt(sm);
    clip_near_zero(norms);
    return tss / af::tile(norms, tss.dims(0));
}  

af::array gauss::normalization::medianNorm(const af::array &tss) {
    auto medians = af::median(tss);
    clip_near_zero(medians);
    return tss / af::tile(medians, tss.dims(0));
}  

af::array gauss::normalization::sigmoidNorm(const af::array &tss) {
    return af::sigmoid(tss);
}  

af::array gauss::normalization::tanhNorm(const af::array &tss) {
    return af::tanh(tss);
}  
