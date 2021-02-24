#include <algos/normalization.h>

af::array algos::normalization::decimalScalingNorm(const af::array &tss) {
    auto maxAbs = af::max(af::abs(tss), 0);
    auto const10 = af::constant(10, 1, tss.dims(1));
    auto d = af::ceil(af::log10(maxAbs));
    auto divFactor = af::pow(const10, d);
    auto result = tss / af::tile(divFactor, static_cast<unsigned int>(tss.dims(0)));
    return result;
}

void algos::normalization::decimalScalingNormInPlace(af::array &tss) {
    auto maxAbs = af::max(af::abs(tss), 0);
    auto const10 = af::constant(10, 1, tss.dims(1));
    auto d = af::ceil(af::log10(maxAbs));
    auto divFactor = af::pow(const10, d);
    tss /= af::tile(divFactor, static_cast<unsigned int>(tss.dims(0)));
}

af::array algos::normalization::maxMinNorm(const af::array &tss, double high, double low, double epsilon) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto scale = max - min;
    auto lessThanEpsilon = epsilon >= scale;
    scale = lessThanEpsilon * lessThanEpsilon.as(tss.type()) + !lessThanEpsilon * scale;
    return low + (((high - low) * (tss - min)) / scale);
}

void algos::normalization::maxMinNormInPlace(af::array &tss, double high, double low, double epsilon) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto scale = max - min;
    auto lessThanEpsilon = epsilon >= scale;
    scale = lessThanEpsilon * lessThanEpsilon.as(tss.type()) + !lessThanEpsilon * scale;
    tss -= min;
    tss *= (high - low);
    tss /= scale;
    tss += low;
}

af::array algos::normalization::meanNorm(const af::array &tss) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto divider = max - min;
    auto dividing = tss - mean;
    return dividing / divider;
}

void algos::normalization::meanNormInPlace(af::array &tss) {
    auto max = af::tile(af::max(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto min = af::tile(af::min(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto divider = max - min;
    tss = tss - mean;
    tss = tss / divider;
}

af::array algos::normalization::znorm(const af::array &tss, double epsilon) {
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto stdev = af::stdev(tss, 0);
    auto lessThanEpsilon = epsilon >= stdev;
    stdev = af::tile(lessThanEpsilon * lessThanEpsilon.as(tss.type()) + !lessThanEpsilon * stdev,
                     static_cast<unsigned int>(tss.dims(0)));
    return (tss - mean) / stdev;
}

void algos::normalization::znormInPlace(af::array &tss, double epsilon) {
    auto mean = af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)));
    auto stdev = af::stdev(tss, 0);
    auto lessThanEpsilon = epsilon >= stdev;
    stdev = af::tile(lessThanEpsilon * lessThanEpsilon.as(tss.type()) + !lessThanEpsilon * stdev,
                     static_cast<unsigned int>(tss.dims(0)));
    tss -= mean;
    tss /= stdev;
}
