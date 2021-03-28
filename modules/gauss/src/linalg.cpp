#include <gauss/linalg.h>

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