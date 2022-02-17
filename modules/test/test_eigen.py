# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets_compute.compute as sc
import numpy as np


def test_eigen_backends():
    for b in sc.get_available_backends():
        sc.set_backend(b)
        data64 = sc.random.randn((3, 3), "float64")
        data32 = data64.astype("float32")
        r64 = sc.eig(data64)
        r32 = sc.eig(data32)
        assert r64.values.same_as(r32.values)
        assert r64.vectors.same_as(r32.vectors)


def test_eigen_h_same_as_np():
    def inner(a):
        assert sc.eigvalsh(a).same_as(np.linalg.eigvalsh(a))
        (v, vec) = sc.eigh(a)
        (nv, nvec) = np.linalg.eigh(a)
        assert v.same_as(nv)

        # changes in sign are expected
        for i in range(a.shape[1]):
            if not vec[:, i].same_as(nvec[:, i]):
                assert vec[:, i].same_as(-1.0 * nvec[:, i])

    inner(np.array([[1, -2j], [2j, 5]]))
    inner(np.array([[5 + 2j, 9 - 2j], [0 + 2j, 2 - 1j]]))


def test_eigen_same_as_np():
    def inner(a):
        assert sc.eigvals(a).same_as(np.linalg.eigvals(a))
        (v, vec) = sc.eig(a)
        (nv, nvec) = np.linalg.eig(a)
        assert v.same_as(nv)
        # changes in sign are expected
        for i in range(a.shape[1]):
            if not vec[:, i].same_as(nvec[:, i]):
                assert vec[:, i].same_as(-1.0 * nvec[:, i])

    inner(np.array([[1, -1], [1, 1]]))
