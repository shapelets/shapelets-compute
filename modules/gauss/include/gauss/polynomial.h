/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_POLYNOMIAL_H
#define GAUSS_POLYNOMIAL_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::polynomial {

/**
 * @brief Least squares polynomial fit. Fit a polynomial \f$p(x) = p[0] * x^{deg} + ... + p[deg]\f$ of degree \f$deg\f$
 * to points \f$(x, y)\f$. Returns a vector of coefficients \f$p\f$ that minimizes the squared error.
 *
 * @param x x-coordinates of the M sample points \f$(x[i], y[i])\f$.
 * @param y y-coordinates of the sample points.
 * @param deg Degree of the fitting polynomial.
 *
 * @return af::array Polynomial coefficients, highest power first.
 */
GAUSSAPI af::array polyfit(const af::array &x, const af::array &y, int deg);

/**
 * @brief Calculates the roots of a polynomial with coefficients given in \f$p\f$. The values in the rank-1 array
 * \f$p\f$ are coefficients of a polynomial. If the length of \f$p\f$ is \f$n+1\f$ then the polynomial is described by:
 * \f[
 *      p[0] * x^n + p[1] * x^{n-1} + ... + p[n-1] * x + p[n]
 * \f]
 *
 * @param pp Array of polynomial coefficients.
 *
 * @return af::array Containing the roots of the polynomial.
 */
GAUSSAPI af::array roots(const af::array &pp);

}  // namespace gauss

#endif
