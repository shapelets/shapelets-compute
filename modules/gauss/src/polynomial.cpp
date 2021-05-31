/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/internal/scopedHostPtr.h>
#include <gauss/linalg.h>
#include <gauss/polynomial.h>

#include <Eigen/Eigenvalues>

using namespace Eigen;

constexpr auto EPSILON = 1e-8;

namespace {

af::array vandermonde(const af::array &x, int order, bool ascending) {
    af::array result = af::array(x.dims(0), order, x.type());
    gfor(af::seq i, order) { result(af::span, i) = af::pow(x, i); }
    if (!ascending) {
        result = af::flip(result, 1);
    }
    return result;
}

}  // namespace

af::array gauss::polynomial::polyfit(const af::array &x, const af::array &y, int deg) {
    int order = deg + 1;
    af::array lhs = vandermonde(x, order, false);
    const af::array &rhs = y;

    af::array scale = af::max(af::sqrt(af::sum(lhs * lhs, 0)), EPSILON);

    lhs /= af::tile(scale, static_cast<unsigned int>(lhs.dims(0)));

    af::array c = gauss::linalg::lls(lhs, rhs);
    c = af::transpose(c);
    c /= af::tile(scale, static_cast<unsigned int>(c.dims(0)));
    c = af::transpose(c);

    return c;
}

af::array gauss::polynomial::roots(const af::array &pp) {
    af::array result = af::array(pp.dims(0) - 1, pp.dims(1), af::dtype::c32);
    for (int i = 0; i < pp.dims(1); i++) {
        af::array p = pp(af::span, i);
        // Strip leading and trailing zeros
        p = p(p != 0);

        p = (-1 * p(af::seq(1, static_cast<double>(p.dims(0)) - 1), af::span)) /
            af::tile(p(0, af::span), static_cast<unsigned int>(p.dims(0)) - 1);

        auto coeffs = gauss::utils::makeScopedHostPtr(p.as(af::dtype::f32).host<float>());

        Eigen::VectorXf vec = Eigen::VectorXf::Ones(p.dims(0));
        Eigen::MatrixXf diag = vec.asDiagonal();

        Eigen::MatrixXf diag2(diag.rows(), diag.cols());
        int rest = static_cast<int>(diag.rows()) - 1;
        diag2.topRows(1) = Eigen::Map<Eigen::MatrixXf>(coeffs.get(), 1, p.dims(0));
        diag2.bottomRows(rest) = diag.topRows(rest);

        Eigen::VectorXcf eivals = diag2.eigenvalues();

        result(af::span, i) = af::array(p.dims(0), (af::cfloat *)eivals.data());
    }

    return result;
}
