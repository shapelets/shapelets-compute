/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/linalg.h>
#include <Eigen/Eigenvalues>
#include <gauss/internal/scopedHostPtr.h>
#include <tuple>

af::array gauss::linalg::eigvalsh(const af::array &m) {
    auto mtype = m.type();

    if (mtype == af::dtype::c32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cfloat>());
        auto typed = (std::complex<float>*)matHost.get();
        Eigen::MatrixXcf mat = Eigen::Map<Eigen::MatrixXcf>(typed, m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        return af::array(m.dims(0), solver.eigenvalues().data());
    }
    
    if (mtype == af::dtype::c64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cdouble>());
        auto typed = (std::complex<double>*)matHost.get();
        Eigen::MatrixXcd mat = Eigen::Map<Eigen::MatrixXcd>(typed, m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        return af::array(m.dims(0), solver.eigenvalues().data());
    }
    
    if (mtype == af::dtype::f64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<double>());
        Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        return af::array(m.dims(0), solver.eigenvalues().data());
    } 
    
    if (mtype == af::dtype::f32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<float>());
        Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        return af::array(m.dims(0), solver.eigenvalues().data());
    }
    
    if (mtype == af::dtype::s64 || mtype == af::dtype::u64) 
        return eigvalsh(m.as(af::dtype::f64));

    return eigvalsh(m.as(af::dtype::f32));
}

af::array gauss::linalg::eigvals(const af::array &m) {
    auto mtype = m.type();

    if (mtype == af::dtype::c32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cfloat>());
        auto typed = (std::complex<float>*)matHost.get();
        Eigen::MatrixXcf mat = Eigen::Map<Eigen::MatrixXcf>(typed, m.dims(0), m.dims(1));
        Eigen::ComplexEigenSolver<Eigen::MatrixXcf> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        auto eivals = solver.eigenvalues();
        auto eivals_af = (af::af_cfloat*) eivals.data();
        return af::array(m.dims(0), eivals_af);
    }
    
    if (mtype == af::dtype::c64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cdouble>());
        auto typed = (std::complex<double>*)matHost.get();
        Eigen::MatrixXcd mat = Eigen::Map<Eigen::MatrixXcd>(typed, m.dims(0), m.dims(1));
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);

        auto eivals = solver.eigenvalues();
        auto eivals_af = (af::af_cdouble*) eivals.data();
        return af::array(m.dims(0), eivals_af);
    }
    
    if (mtype == af::dtype::f64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<double>());
        Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::EigenSolver<Eigen::MatrixXd> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        auto eivals = solver.eigenvalues();
        auto eivals_af = (af::af_cdouble*) eivals.data();
        return af::array(m.dims(0), eivals_af);
    } 
    
    if (mtype == af::dtype::f32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<float>());
        Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::EigenSolver<Eigen::MatrixXf> solver(mat, Eigen::DecompositionOptions::EigenvaluesOnly);
        auto eivals = solver.eigenvalues();
        auto eivals_af = (af::af_cfloat*) eivals.data();
        return af::array(m.dims(0), eivals_af);
    }
    
    if (mtype == af::dtype::s64 || mtype == af::dtype::u64) 
        return eigvals(m.as(af::dtype::f64));

    return eigvals(m.as(af::dtype::f32));
}

std::tuple<af::array, af::array> gauss::linalg::eig(const af::array &m) {

    auto mtype = m.type();

    if (mtype == af::dtype::c32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cfloat>());
        auto typed = (std::complex<float>*)matHost.get();
        Eigen::MatrixXcf mat = Eigen::Map<Eigen::MatrixXcf>(typed, m.dims(0), m.dims(1));
        Eigen::ComplexEigenSolver<Eigen::MatrixXcf> solver(mat);
        auto eivals = solver.eigenvalues();
        auto eivect = solver.eigenvectors();
        auto eivals_af = (af::af_cfloat*) eivals.data();
        auto eivect_af = (af::af_cfloat*) eivect.data();
        auto eigenValues = af::array(m.dims(0), eivals_af);
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af);
        return std::make_tuple(eigenValues, eigenVectors);
    }
    
    if (mtype == af::dtype::c64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cdouble>());
        auto typed = (std::complex<double>*)matHost.get();

        Eigen::MatrixXcd mat = Eigen::Map<Eigen::MatrixXcd>(typed, m.dims(0), m.dims(1));
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(mat);

        auto eivals = solver.eigenvalues();
        auto eivect = solver.eigenvectors();
        auto eivals_af = (af::af_cdouble*) eivals.data();
        auto eivect_af = (af::af_cdouble*) eivect.data();
        auto eigenValues = af::array(m.dims(0), eivals_af);
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af);       
        return std::make_tuple(eigenValues, eigenVectors); 
    }
    
    if (mtype == af::dtype::f64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<double>());
        Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::EigenSolver<Eigen::MatrixXd> solver(mat);
        auto eivals = solver.eigenvalues();
        auto eivect = solver.eigenvectors();
        auto eivals_af = (af::af_cdouble*) eivals.data();
        auto eivect_af = (af::af_cdouble*) eivect.data();
        auto eigenValues = af::array(m.dims(0), eivals_af);
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af);     
        return std::make_tuple(eigenValues, eigenVectors);
    } 
    
    if (mtype == af::dtype::f32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<float>());
        Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::EigenSolver<Eigen::MatrixXf> solver(mat);
        auto eivals = solver.eigenvalues();
        auto eivect = solver.eigenvectors();
        auto eivals_af = (af::af_cfloat*) eivals.data();
        auto eivect_af = (af::af_cfloat*) eivect.data();
        auto eigenValues = af::array(m.dims(0), eivals_af);
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af); 
        return std::make_tuple(eigenValues, eigenVectors);
    }
    
    if (mtype == af::dtype::s64 || mtype == af::dtype::u64) 
        return eig(m.as(af::dtype::f64));

    return eig(m.as(af::dtype::f32));
}


std::tuple<af::array, af::array> gauss::linalg::eigh(const af::array &m) {

    auto mtype = m.type();

    if (mtype == af::dtype::c32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cfloat>());
        auto typed = (std::complex<float>*)matHost.get();
        Eigen::MatrixXcf mat = Eigen::Map<Eigen::MatrixXcf>(typed, m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> solver(mat);
        auto eigenValues = af::array(m.dims(0), solver.eigenvalues().data());

        auto eivect = solver.eigenvectors();
        auto eivect_af = (af::af_cfloat*) eivect.data();
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af);       

        return std::make_tuple(eigenValues, eigenVectors); 
    }
    
    if (mtype == af::dtype::c64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<af::af_cdouble>());
        auto typed = (std::complex<double>*)matHost.get();

        Eigen::MatrixXcd mat = Eigen::Map<Eigen::MatrixXcd>(typed, m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(mat);

        auto eigenValues = af::array(m.dims(0), solver.eigenvalues().data());

        auto eivect = solver.eigenvectors();
        auto eivect_af = (af::af_cdouble*) eivect.data();
        auto eigenVectors = af::array(m.dims(0), m.dims(1), eivect_af);       

        return std::make_tuple(eigenValues, eigenVectors); 
    }
    
    if (mtype == af::dtype::f64) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<double>());
        Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);
        auto eigenValues = af::array(m.dims(0), solver.eigenvalues().data());
        auto eigenVectors = af::array(m.dims(0), m.dims(1), solver.eigenvectors().data());     
        return std::make_tuple(eigenValues, eigenVectors);
    } 
    
    if (mtype == af::dtype::f32) {
        auto matHost = gauss::utils::makeScopedHostPtr(m.host<float>());
        Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost.get(), m.dims(0), m.dims(1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(mat);

        auto eigenValues = af::array(m.dims(0), solver.eigenvalues().data());
        auto eigenVectors = af::array(m.dims(0), m.dims(1), solver.eigenvectors().data()); 
        return std::make_tuple(eigenValues, eigenVectors);
    }
    
    if (mtype == af::dtype::s64 || mtype == af::dtype::u64) 
        return eigh(m.as(af::dtype::f64));

    return eigh(m.as(af::dtype::f32));
}


af::array gauss::linalg::lls(const af::array &A, const af::array &b) {
    af::array U;
    af::array S;
    af::array VT;

    af::svd(U, S, VT, A);

    S = af::diag(S, 0, false);

    af::array S_dagger = (S != 0).as(S.type()) * af::inverse(S);

    long missingRows = static_cast<long>(A.dims(1) - S.dims(0));
    long missingColumns = static_cast<long>(A.dims(0) - S.dims(0));
    af::array toPadRows = af::constant(0, missingRows, A.dims(0), A.type());
    af::array toPadColumns = af::constant(0, A.dims(1), missingColumns, A.type());

    S_dagger = af::join(0, S_dagger, toPadRows);
    S_dagger = af::join(1, S_dagger, toPadColumns);

    af::array V = af::transpose(VT);
    af::array UT = af::transpose(U);
    af::array x = af::matmul(V, S_dagger, UT, b);

    return x;
}


af::array gauss::linalg::levinsonDurbin(af::array acv, int order) {
    af::array result = af::constant(0, order + 1, acv.dims(1), acv.type());

    for (int i = 0; i < acv.dims(1); i++) {
        af::array phi = af::constant(0, order + 1, order + 1, acv.type());
        af::array sig = af::constant(0, order + 1, acv.type());

        phi(1, 1) = acv(1, i) / acv(0, i);
        sig(1) = acv(0, i) - (phi(1, 1) * acv(1, i));

        // First iteration, to avoid problems with negative sequences with 1 element
        int k = 2;
        if (k < (order + 1)) {
            phi(k, k) = (acv(k, i) - af::dot(phi(af::seq(1, k - 1), k - 1), acv(af::seq(1, k - 1), i))) / sig(k - 1);
            for (int j = 1; j < k; j++) {
                phi(j, k) = phi(j, k - 1) - (phi(k, k) * phi(k - j, k - 1));
            }
            sig(k) = sig(k - 1) * (1.0 - phi(k, k) * phi(k, k));
        }

        // Second and subsequent iterations
        for (int l = 3; l < (order + 1); l++) {
            af::array aux = acv(af::seq(1, l - 1), i);
            phi(l, l) = (acv(l, i) - af::dot(phi(af::seq(1, l - 1), l - 1),
                                             aux(af::seq(static_cast<double>(aux.dims(0)) - 1, 0, -1)))) /
                        sig(l - 1);
            for (int j = 1; j < l; j++) {
                phi(j, l) = phi(j, l - 1) - (phi(l, l) * phi(l - j, l - 1));
            }
            sig(l) = sig(l - 1) * (1.0 - phi(l, l) * phi(l, l));
        }

        af::array pac = af::diag(phi);
        pac(0) = 1.0;
        result(af::span, i) = pac;
    }
    return result;
}