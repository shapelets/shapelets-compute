# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import numpy as np
import shapelets.compute as sc


def test_sp_1d_cases():
    for b in sc.get_available_backends():
        a = np.array([0., 1, 2, 3, 4, 3, 2, 1, 0, 1], dtype="float32")
        assert sc.fft.fft(a).same_as(np.fft.fft(a))
        # assert sc.fft.fft(a, norm="forward").same_as(np.fft.fft(a, norm="forward"))
        assert sc.fft.fft(a, norm="ortho").same_as(np.fft.fft(a, norm="ortho"))
        # assert sc.fft.fft(a, norm="backward").same_as(np.fft.fft(a, norm="backward"))
        assert sc.fft.fft(a, shape=5).same_as(np.fft.fft(a, n=5))
        assert sc.fft.fft(a, shape=15).same_as(np.fft.fft(a, n=15))

        b = np.fft.fft(a)
        assert sc.fft.ifft(b).same_as(np.fft.ifft(b))
        # assert sc.fft.ifft(b, norm="forward").same_as(np.fft.ifft(b, norm="forward"))
        assert sc.fft.ifft(b, norm="ortho").same_as(np.fft.ifft(b, norm="ortho"))
        # assert sc.fft.ifft(b, norm="backward").same_as(np.fft.ifft(b, norm="backward"))

        b = np.fft.fft(a, n=5)
        assert sc.fft.ifft(b).same_as(np.fft.ifft(b))

        b = np.fft.fft(a, n=50)
        assert sc.fft.ifft(b).same_as(np.fft.ifft(b))

        npa = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3], dtype="float32").reshape(4, 3)
        sha = sc.array(npa)
        assert sc.fft.fft(sha[::, 0]).same_as(np.fft.fft(npa[::, 0]))
        assert sc.fft.fft(sha[::, 1]).same_as(np.fft.fft(npa[::, 1]))
        assert sc.fft.fft(sha[::, 2]).same_as(np.fft.fft(npa[::, 2]))
        assert sc.fft.fft(sha[0, ::]).same_as(np.fft.fft(npa[0, ::]))
        assert sc.fft.fft(sha[1, ::]).same_as(np.fft.fft(npa[1, ::]))
        assert sc.fft.fft(sha[2, ::]).same_as(np.fft.fft(npa[2, ::]))
        assert sc.fft.fft(sha[3, ::]).same_as(np.fft.fft(npa[3, ::]))


def test_sp_convolve():
    for b in sc.get_available_backends():
        sc.set_backend(b)
        a = sc.random.randn(1000, "float32")
        b = sc.random.randn(10, "float32")
        r = sc.convolve1(a, b, 'expand', 'frequency')
        a64 = a.astype("float64")
        b64 = b.astype("float64")
        r64 = sc.convolve1(a64, b64, 'expand', 'frequency')
        assert r64.same_as(r.astype("float64"))
