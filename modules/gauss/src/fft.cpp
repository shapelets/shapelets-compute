#include <gauss/fft.h>
#include <variant>
#include <optional>

namespace gauss::fft {

inline double towardsFrequencyDomain(Norm norm, int32_t n) {
    switch (norm)
    {
        case Norm::Backward: return 1.0;
        case Norm::Orthonormal: return 1.0 / sqrt(static_cast<double>(n));
        default: return 1.0 / static_cast<double>(n);
    }
}

inline double towardsTimeDomain(Norm norm, int32_t n) {
    switch (norm)
    {
        case Norm::Backward: return 1.0 / static_cast<double>(n);
        case Norm::Orthonormal: return 1.0 / sqrt(static_cast<double>(n));
        default: return 1.0;
    }
}

af::array fft(const af::array& signal) {
    return fft(signal, signal.dims());
}

af::array fft(const af::array& signal, const af::dim4& outDims) {
    return fft(signal, Norm::Backward, outDims);
}

af::array ifft(const af::array& coef) {
    return ifft(coef, coef.dims());
}

af::array ifft(const af::array& coef, const af::dim4& outDims) {
    return ifft(coef, Norm::Backward, outDims);
}

af::array fft(const af::array& signal, const std::variant<Norm, double>& norm, const af::dim4& outDims, double* actualNorm) {
    auto dims = outDims.ndims();

    double computed_norm;
    if(auto pnorm = std::get_if<Norm>(&norm)) {
        int32_t n;
        if (dims == 1) 
            n = outDims[0]; 
        else if (dims == 2) 
            n = outDims[0] * outDims[1]; 
        else 
            n = outDims[0] * outDims[1] * outDims[2]; 
        
        computed_norm = towardsFrequencyDomain(*pnorm, n);
    } 
    else {
        computed_norm = std::get<double>(norm);
    }

    if (actualNorm != nullptr)
        *actualNorm = computed_norm;

    switch(dims) {
        case 1: return af::fftNorm(signal, computed_norm, outDims[0]);
        case 2: return af::fft2Norm(signal, computed_norm, outDims[0], outDims[1]);
        default: return af::fft3Norm(signal, computed_norm, outDims[0], outDims[1], outDims[2]);
    }
}

af::array ifft(const af::array& coef, const std::variant<Norm, double> norm, const af::dim4& outDims, double* actualNorm) {
    auto dims = outDims.ndims();
    double computed_norm;

    if(auto pnorm = std::get_if<Norm>(&norm)) {
        int32_t n;
        if (dims == 1) 
            n = outDims[0]; 
        else if (dims == 2) 
            n = outDims[0] * outDims[1]; 
        else 
            n = outDims[0] * outDims[1] * outDims[2]; 
        
        computed_norm = towardsTimeDomain(*pnorm, n);
    } 
    else {
        computed_norm = std::get<double>(norm);
    }

    if (actualNorm != nullptr)
        *actualNorm = computed_norm;

    switch(dims) {
        case 1: return af::ifftNorm(coef, computed_norm, outDims[0]);
        case 2: return af::ifft2Norm(coef, computed_norm, outDims[0], outDims[1]);
        default: return af::ifft3Norm(coef, computed_norm, outDims[0], outDims[1], outDims[2]);
    }
}

af::array rfft(const af::array& signal) {
    return rfft(signal, signal.dims());
}

af::array rfft(const af::array& signal, const af::dim4& outDims) {
    return rfft(signal, Norm::Backward, outDims);
}

af::array rfft(const af::array& signal, const std::variant<Norm, double> norm, const af::dim4& outDims) {
    auto dims = outDims.ndims();

    double computed_norm;
    if(auto pnorm = std::get_if<Norm>(&norm)) {
        int32_t n;
        if (dims == 1) 
            n = outDims[0]; 
        else if (dims == 2) 
            n = outDims[0] * outDims[1]; 
        else 
            n = outDims[0] * outDims[1] * outDims[2]; 

        computed_norm = towardsFrequencyDomain(*pnorm, n);
    } 
    else {
        computed_norm = std::get<double>(norm);
    }

    switch(dims) {
        case 1: 
            return af::fftR2C<1>(signal, outDims, computed_norm);
        case 2: 
            return af::fftR2C<2>(signal, outDims, computed_norm);
        default: 
            return af::fftR2C<3>(signal, outDims, computed_norm);
    }
}

af::array irfft(const af::array& coef, const bool is_odd) {
    auto newDims = af::dim4(coef.dims());
    newDims[0] = newDims[0] * 2 + (is_odd ? 1 : 0);
    return irfft(coef, newDims);
}

af::array irfft(const af::array& coef, const af::dim4& outDims) {
    return irfft(coef, Norm::Backward, outDims);
}


af::array irfft(const af::array& coef, const std::variant<Norm, double> norm, const af::dim4& outDims) {
    auto dims = outDims.ndims();

    double computed_norm;
    if(auto pnorm = std::get_if<Norm>(&norm)) {
        int32_t n;
        if (dims == 1) 
            n = outDims[0]; 
        else if (dims == 2) 
            n = outDims[0] * outDims[1]; 
        else 
            n = outDims[0] * outDims[1] * outDims[2]; 
        computed_norm = towardsTimeDomain(*pnorm, n);
    } 
    else {
        computed_norm = std::get<double>(norm);
    }

    switch(dims) {
        case 1: 
            return af::fftC2R<1>(coef, outDims[0] % 2 == 1, computed_norm);
        case 2: 
            return af::fftC2R<2>(coef, outDims[0] % 2 == 1, computed_norm);
        default: 
            return af::fftC2R<3>(coef, outDims[0] % 2 == 1, computed_norm);
    }
}


af::array rfftfreq(const int32_t n, const double d) {
    double v = 1.0 / (n * d);
    auto N = (n >> 1) + 1;
    auto r = af::range(af::dim4(N));
    return r * v;
}

af::array fftfreq(const unsigned int n, const double d) {
    double v = 1.0 / (n * d);
    auto N = static_cast<long long>(((n - 1) >> 1) + 1);
    auto r1 = af::range(af::dim4(N));
    af::array r2 = af::seq(-(n >> 1), -1);
    return af::join(0, r1, r2.as(r1.type())) * v;
}

af::array fftshift(const af::array& x, const std::optional<std::variant<int, std::vector<int>>>& axes) {
    // just in case
    if (x.isempty())
        return x;

    // the shifts we end up doing
    int shifts[4] = {0, 0, 0, 0};
    
    if (!axes) {
        // no data provided
        // iterate through all dimensions and shift by 1/2
        for (auto i = 0u; i<x.numdims(); i++) {
            shifts[i] = x.dims(i) >> 1;
        }
    }
    else {
        
        auto var = axes.value();
        if (var.index() == 0) {
            // just one axis provided
            auto dim = std::get<int>(var);
            if (dim >= 4)
                throw std::invalid_argument("No more than four dimensions are applicable.");
            shifts[dim] = x.dims(dim) >> 1; 
        }
        else {
            // concrete axes provided
            auto dims = std::get<std::vector<int>>(var);
            for (auto dim: dims) {
                if (dim >= 4)
                    throw std::invalid_argument("No more than four dimensions are applicable.");
                shifts[dim] = x.dims(dim) >> 1; 
            }
        }
    }

    return af::shift(x, shifts[0], shifts[1], shifts[2], shifts[3]);
}


af::array spectral_derivative(const af::array& signal, const std::variant<double, af::array> kappa_spec, const bool shift) {
    // number of points
    auto n = signal.dims(0);
    // peform the fft 
    auto fhat = af::fft(signal);
    
    af::array kappa;
    if (kappa_spec.index()==0) {
        // compute kappa (spatial frequencies, aka wave numbers)
        auto domain_length = std::get<double>(kappa_spec);
        auto unshifted_kappa = (2.0 * af::Pi / domain_length) * (af::range(n) - (n>>1));
        // shift the spatial frequencies so the match the Fourier transform terms.
        kappa = fftshift(unshifted_kappa);
    }
    else {
        auto given_kappa = std::get<af::array>(kappa_spec);
        if (given_kappa.dims(0) != n)
            throw std::invalid_argument("Kappa specification must have the same lenght as the signal");
        // shift if requested
        kappa = (shift) ? fftshift(given_kappa) : given_kappa;
    }

    // compute i*k*f --> which gives the fourier of the derivative
    auto dfhat = kappa * fhat * af::cfloat(0.0, 1.0);
    // put it back
    auto df = af::ifft(dfhat);
    // remove the imaginary part of the result.
    return af::real(df);
}

}