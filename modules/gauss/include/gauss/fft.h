/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_FFT_H
#define GAUSS_FFT_H

#include <arrayfire.h>
#include <gauss/defines.h>
#include <variant>
#include <optional>

namespace gauss::fft {

    enum class Norm { Backward, Orthonormal, Forward };

    /**
     * @brief Computes the spectral derivative of a signal
     * 
     * @param signal This is a column vector denoting the points of the signal.  
     * @param kappa_spec This is the specification for building the kappa coeffients.  It could be either 
     * a double number, denoting a domain length to scale the automatic generation of kappa vector from 
     * -n/2 to n/2.  Alternatively, one can pass a column vector of size n for the desired values.
     * @param shift When kappa_spec is a vector, this flag determines if it is required to adjust the values
     * of the vector by calling `fftshift`.  Defaults to `true`.
     */ 
    af::array spectral_derivative(const af::array& signal, const std::variant<double, af::array> kappa_spec, const bool shift = true);

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
    GAUSSAPI af::array fftfreq(const unsigned int n, const double d = 1.0);


    GAUSSAPI af::array fftshift(const af::array& x, const std::optional<std::variant<int, std::vector<int>>>& axes = std::nullopt); 

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