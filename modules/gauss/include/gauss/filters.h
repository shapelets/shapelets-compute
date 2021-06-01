/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_FILTERS_H
#define GAUSS_FILTERS_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::filters {

/**
 * @brief Smooth (and optionally differentiate) data with a Savitzky-Golay filter.  
 * This filter is a low pass filter, particularly suited for smoothing noisy data; the 
 * main idea behind this approach is to make for each point a least-square fit with 
 * a polynomical of high order over an odd-sized window centered at the point.
 *
 * @param y signal to process
 * @param window_size the length of the window; must be an odd integer number 
 * @param order the order of the polynomial used in the filtering 
 * @param deriv the order of the derivative to compute; if set to zero, it simply performs a smooth operation.
 *
 * @return af::array the smoothed signal or its nth derivative (as per deriv parameter)
 * 
 * [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
 * Data by Simplified Least Squares Procedures. Analytical
 * Chemistry, 1964, 36 (8), pp 1627-1639.
 * 
 * [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
 * W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
 * Cambridge University Press ISBN-13: 9780521880688
 * 
 * Online References
 *  - https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
 *  - https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay (Used as reference implementation)
 */
 GAUSSAPI af::array savitzkyGolay(const af::array &y, int window_size, int order, int deriv);

}

#endif
