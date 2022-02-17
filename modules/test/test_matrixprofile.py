# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets_compute.compute as sc


def test_matprof_iface():
    tss = sc.cumsum(sc.random.randn((100, 3)), 0)
    r = sc.matrixprofile.matrix_profile(tss, 10)
    assert r.window == 10
    assert r.index.shape == (91, 3)
    assert r.profile.shape == r.index.shape
