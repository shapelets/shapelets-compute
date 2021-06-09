/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_LINALG_H
#define GAUSS_LINALG_H

#include <arrayfire.h>
#include <gauss/defines.h>

namespace gauss::linalg {

/**
 * @brief Calculates the minimum norm least squares solution \f$x\f$ \f$(\left\lVert{A·x - b}\right\rVert^2)\f$ to
 * \f$A·x = b\f$. This function uses the singular value decomposition function of Arrayfire. The actual formula that
 * this function computes is \f$x = V·D\dagger·U^T·b\f$. Where \f$U\f$ and \f$V\f$ are orthogonal matrices and
 * \f$D\dagger\f$ contains the inverse values of the singular values contained in D if they are not zero, and zero
 * otherwise.
 *
 * @param A Coefficient matrix containing the coefficients of the linear equation problem to solve.
 * @param b Vector with the measured values.
 *
 * @return af::array Contains the solution to the linear equation problem minimizing the norm 2.
 */
GAUSSAPI af::array lls(const af::array &A, const af::array &b);


GAUSSAPI af::array levinsonDurbin(af::array acv, int order);

/**
 * @brief Computes the eigen values and vectors of the input array.  General method.
 * @return Tuple of arrays, where the first element corresponds to the eigen values and the second 
 * item are the eigen values.
 */ 
std::tuple<af::array, af::array> eig(const af::array &m);

std::tuple<af::array, af::array> eigh(const af::array &m);

/**
 * @brief Computes the eigen values of the input array.  General method.
 */ 
af::array eigvals(const af::array &m); 

/**
 * @brief Computes the eigen values of the input array.  General method.
 */ 
af::array eigvalsh(const af::array &m); 

}  // namespace gauss

#endif
