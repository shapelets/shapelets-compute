#ifndef GAUSS_FFT_H
#define GAUSS_FFT_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <variant>

namespace gauss::fft {

    enum class Norm { Backward, Orthonormal, Forward };

    /**
     * @brief Return the Discrete Fourier Transform sample frequencies 
     * 
     * The returned float array `f` contains the frequency bin centers in cycles
     * per unit of the sample spacing (with zero at the start).  For instance, 
     * if the sample spacing is in seconds, then the frequency unit is cycles/second.
     * 
     * Given a window length `n` and a sample spacing `d`
     *       f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
     *       f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd
     * 
     * Unlike `fftfreq`, the Nyquist frequency component is considered 
     * to be positive.
     * 
     * @param n Window length.
     * @param d Sample spacing (inverse of the sampling rate). Defaults to 1.
     * @return af::array An array of one column and `n` rows containing the sample 
     * frequencies.
     */ 
    GAUSSAPI af::array rfftfreq(const int32_t n, const double d = 1.0);
    
    
    /**
     * @brief Return the Discrete Fourier Transform sample frequencies.
     * 
     * The returned float array `f` contains the frequency bin centers in cycles
     * per unit of the sample spacing (with zero at the start).  For instance, if
     * the sample spacing is in seconds, then the frequency unit is cycles/second.
     * 
     * Given a window length `n` and a sample spacing `d`
     *       f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
     *       f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
     * 
     * @param n Window length.
     * @param d Sample spacing (inverse of the sampling rate). Defaults to 1.
     * @return af::array An array of one column and `n` rows containing the sample 
     * frequencies.
     */ 
    GAUSSAPI af::array fftfreq(const int32_t n, const double d = 1.0);

    /**
     */ 
    GAUSSAPI af::array fft(const af::array& signal);

    /**
     */ 
    GAUSSAPI af::array fft(const af::array& signal, const af::dim4& outDims);

    /**
     */ 
    GAUSSAPI af::array fft(const af::array& signal, const std::variant<Norm, double>& norm, const af::dim4& outDims, double* actNorm = nullptr); 

    /**
     */ 
    GAUSSAPI af::array ifft(const af::array& coef);
    
    /**
     */ 
    GAUSSAPI af::array ifft(const af::array& coef, const af::dim4& outDims);
    
    /**
     */ 
    GAUSSAPI af::array ifft(const af::array& coef, const std::variant<Norm, double> norm, const af::dim4& outDims, double* actNorm = nullptr);

    /**
     */ 
    GAUSSAPI af::array rfft(const af::array& signal);

    /**
     */ 
    GAUSSAPI af::array rfft(const af::array& signal, const af::dim4& outDims);

    /**
     */ 
    GAUSSAPI af::array rfft(const af::array& signal, const std::variant<Norm, double> norm, const af::dim4& outDims);

    GAUSSAPI af::array irfft(const af::array& coef, const bool is_odd);

    /**
     */ 
    GAUSSAPI af::array irfft(const af::array& coef, const af::dim4& outDims);

    /**
     */ 
    GAUSSAPI af::array irfft(const af::array& coef, const std::variant<Norm, double> norm, const af::dim4& outDims);

}

#endif  //GAUSS_FFT_H