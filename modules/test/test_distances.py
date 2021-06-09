# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets.compute as sc
from shapelets.compute.distances import DistanceType
import os
import numpy as np


def test_dist_euclidean():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.euclidean(a, b).same_as([
        [1.0000, 0.0000, 1.4142, 1.0000, 1.4142, 1.0000, 1.7321, 1.4142],
        [1.0000, 1.4142, 0.0000, 1.0000, 1.4142, 1.7321, 1.0000, 1.4142],
        [1.0000, 1.4142, 1.4142, 1.7321, 0.0000, 1.0000, 1.0000, 1.4142]
    ])

    assert sc.distances.pdist(b, 'euclidean').same_as([
        [0.0000, 1.0000, 1.0000, 1.4142, 1.0000, 1.4142, 1.4142, 1.7321],
        [1.0000, 0.0000, 1.4142, 1.0000, 1.4142, 1.0000, 1.7321, 1.4142],
        [1.0000, 1.4142, 0.0000, 1.0000, 1.4142, 1.7321, 1.0000, 1.4142],
        [1.4142, 1.0000, 1.0000, 0.0000, 1.7321, 1.4142, 1.4142, 1.0000],
        [1.0000, 1.4142, 1.4142, 1.7321, 0.0000, 1.0000, 1.0000, 1.4142],
        [1.4142, 1.0000, 1.7321, 1.4142, 1.0000, 0.0000, 1.4142, 1.0000],
        [1.4142, 1.7321, 1.0000, 1.4142, 1.0000, 1.4142, 0.0000, 1.0000],
        [1.7321, 1.4142, 1.4142, 1.0000, 1.4142, 1.0000, 1.0000, 0.0000]
    ])


def test_dist_manhattan():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0.0, 0, 1, 1, 1, 1],
        [0, 0, 1.3, 1, 0, 0, 1, 1],
        [0, 1, 0.0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.manhattan(a, b).same_as([
        [1.0000, 0.0000, 2.3000, 1.0000, 2.0000, 1.0000, 3.0000, 2.0000],
        [1.0000, 2.0000, 0.3000, 1.0000, 2.0000, 3.0000, 1.0000, 2.0000],
        [1.0000, 2.0000, 2.3000, 3.0000, 0.0000, 1.0000, 1.0000, 2.0000],
    ])

    assert sc.distances.pdist(b, 'manhattan').same_as([
        [0.0000, 1.0000, 1.3000, 2.0000, 1.0000, 2.0000, 2.0000, 3.0000],
        [1.0000, 0.0000, 2.3000, 1.0000, 2.0000, 1.0000, 3.0000, 2.0000],
        [1.3000, 2.3000, 0.0000, 1.3000, 2.3000, 3.3000, 1.3000, 2.3000],
        [2.0000, 1.0000, 1.3000, 0.0000, 3.0000, 2.0000, 2.0000, 1.0000],
        [1.0000, 2.0000, 2.3000, 3.0000, 0.0000, 1.0000, 1.0000, 2.0000],
        [2.0000, 1.0000, 3.3000, 2.0000, 1.0000, 0.0000, 2.0000, 1.0000],
        [2.0000, 3.0000, 1.3000, 2.0000, 1.0000, 2.0000, 0.0000, 1.0000],
        [3.0000, 2.0000, 2.3000, 1.0000, 2.0000, 1.0000, 1.0000, 0.0000]
    ])


def test_dist_hamming():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.hamming(a, b).same_as([
        [1, 0, 2, 1, 2, 1, 3, 2],
        [1, 2, 0, 1, 2, 3, 1, 2],
        [1, 2, 2, 3, 0, 1, 1, 2]
    ])
    assert sc.distances.pdist(b, 'hamming').same_as([
        [0, 1, 1, 2, 1, 2, 2, 3],
        [1, 0, 2, 1, 2, 1, 3, 2],
        [1, 2, 0, 1, 2, 3, 1, 2],
        [2, 1, 1, 0, 3, 2, 2, 1],
        [1, 2, 2, 3, 0, 1, 1, 2],
        [2, 1, 3, 2, 1, 0, 2, 1],
        [2, 3, 1, 2, 1, 2, 0, 1],
        [3, 2, 2, 1, 2, 1, 1, 0],
    ])


def test_dist_minkowski():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.minkowski(a, b, 2.0).same_as(
        sc.distances.euclidean(a, b))
    assert sc.distances.minkowski(a, b, 1.0).same_as(
        sc.distances.manhattan(a, b))
    assert sc.distances.minkowski(a, b, 0.5).same_as([
        [1.0000, 0.0000, 4.0000, 1.0000, 4.0000, 1.0000, 9.0000, 4.0000],
        [1.0000, 4.0000, 0.0000, 1.0000, 4.0000, 9.0000, 1.0000, 4.0000],
        [1.0000, 4.0000, 4.0000, 9.0000, 0.0000, 1.0000, 1.0000, 4.0000]
    ])

    assert sc.distances.pdist(b, 'minkowski', p=0.5).same_as([
        [0.0000, 1.0000, 1.0000, 4.0000, 1.0000, 4.0000, 4.0000, 9.0000],
        [1.0000, 0.0000, 4.0000, 1.0000, 4.0000, 1.0000, 9.0000, 4.0000],
        [1.0000, 4.0000, 0.0000, 1.0000, 4.0000, 9.0000, 1.0000, 4.0000],
        [4.0000, 1.0000, 1.0000, 0.0000, 9.0000, 4.0000, 4.0000, 1.0000],
        [1.0000, 4.0000, 4.0000, 9.0000, 0.0000, 1.0000, 1.0000, 4.0000],
        [4.0000, 1.0000, 9.0000, 4.0000, 1.0000, 0.0000, 4.0000, 1.0000],
        [4.0000, 9.0000, 1.0000, 4.0000, 1.0000, 4.0000, 0.0000, 1.0000],
        [9.0000, 4.0000, 4.0000, 1.0000, 4.0000, 1.0000, 1.0000, 0.0000]
    ])


def test_dist_chebyshev():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.chebyshev(a, b).same_as([
        [1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000]
    ])

    assert sc.distances.pdist(b, 'chebyshev').same_as([
        [0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000]
    ])


def test_dist_sbd():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.sbd(a, b).same_as([
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226],
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226],
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226]
    ])

    assert sc.distances.pdist(b, 'sbd').same_as([
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226],
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226],
        [0.2929, 0.2929, 0.0000, 0.2929, 0.5000, -0.0000, 0.1835],
        [0.0000, 0.0000, 0.2929, 0.0000, 0.2929, 0.2929, 0.4226],
        [0.2929, 0.2929, 0.5000, 0.2929, 0.0000, 0.5000, 0.1835],
        [0.2929, 0.2929, -0.0000, 0.2929, 0.5000, 0.0000, 0.1835],
        [0.4226, 0.4226, 0.1835, 0.4226, 0.1835, 0.1835, 0.0000]
    ])


def test_dist_dtw():
    a = sc.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype="float32")

    b = sc.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ], dtype="float32")

    assert sc.distances.dtw(a, b).same_as([
        [1.0000, 0.0000, 1.0000, 0.0000, 2.0000, 1.0000, 3.0000, 2.0000],
        [1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 2.0000, 1.0000, 2.0000],
        [1.0000, 2.0000, 1.0000, 3.0000, 0.0000, 1.0000, 0.0000, 2.0000]
    ])

    assert sc.distances.pdist(b, 'dtw').same_as([
        [0.0000, 1.0000, 1.0000, 2.0000, 1.0000, 2.0000, 2.0000, 3.0000],
        [1.0000, 0.0000, 1.0000, 0.0000, 2.0000, 1.0000, 3.0000, 2.0000],
        [1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 2.0000, 1.0000, 2.0000],
        [2.0000, 0.0000, 1.0000, 0.0000, 3.0000, 1.0000, 2.0000, 1.0000],
        [1.0000, 2.0000, 1.0000, 3.0000, 0.0000, 1.0000, 0.0000, 2.0000],
        [2.0000, 1.0000, 2.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000],
        [2.0000, 3.0000, 1.0000, 2.0000, 0.0000, 1.0000, 0.0000, 1.0000],
        [3.0000, 2.0000, 2.0000, 1.0000, 2.0000, 1.0000, 1.0000, 0.0000]
    ])


def test_dist_mpdist():
    ts = sc.array([1., 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2, 4, 5, 1, 1, 9], dtype="float64")
    query = sc.array([0.23595094, 0.9865171, 0.1934413, 0.60880883, 0.55174926, 0.77139988, 0.33529215, 0.63215848],
                     dtype="float64")
    s1 = sc.distances.mpdist(ts, query, 4)
    s2 = sc.distances.mpdist(query, ts, 4)
    assert s1.same_as(s2)
    assert s1.same_as([0.4377])


def test_dist_mpdist_against_data():
    import pathlib
    current_path = pathlib.Path(__file__).parent.absolute()
    ts = np.loadtxt(os.path.join(current_path, 'resources/sampledata.txt'))
    tsb = ts[199:300]
    w = 32
    assert sc.distances.mpdist(ts, tsb, w).same_as([0.])


def test_dist_should_not_throw():
    a = sc.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype="float32")
    b = sc.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], dtype="float32")
    sc.distances.euclidean(a, b)
    sc.distances.manhattan(a, b)
    sc.distances.minkowski(a, b, 2)
    sc.distances.chebyshev(a, b)
    sc.distances.sorensen(a, b)
    sc.distances.gower(a, b)
    sc.distances.soergel(a, b)
    sc.distances.kulczynski(a, b)
    sc.distances.canberra(a, b)
    sc.distances.lorentzian(a, b)
    sc.distances.intersection(a, b)
    sc.distances.wavehedges(a, b)
    sc.distances.czekanowski(a, b)
    sc.distances.ruzicka(a, b)
    sc.distances.motyka(a, b)
    sc.distances.tanimoto(a, b)
    sc.distances.innerproduct(a, b)
    sc.distances.harmonic_mean(a, b)
    sc.distances.cosine(a, b)
    sc.distances.kumarhassebrook(a, b)
    sc.distances.jaccard(a, b)
    sc.distances.dice(a, b)
    sc.distances.fidelity(a, b)
    sc.distances.bhattacharyya(a, b)
    sc.distances.hellinger(a, b)
    sc.distances.matusita(a, b)
    sc.distances.square_chord(a, b)
    sc.distances.squared_euclidean(a, b)
    sc.distances.pearson(a, b)
    sc.distances.neyman(a, b)
    sc.distances.squared_chi(a, b)
    sc.distances.prob_symmetric_chi(a, b)
    sc.distances.divergence(a, b)
    sc.distances.clark(a, b)
    sc.distances.additive_symm_chi(a, b)
    sc.distances.kullback(a, b)
    sc.distances.jeffrey(a, b)
    sc.distances.k_divergence(a, b)
    sc.distances.topsoe(a, b)
    sc.distances.jensen_shannon(a, b)
    sc.distances.jensen_difference(a, b)
    sc.distances.taneja(a, b)
    sc.distances.kumar_johnson(a, b)
    sc.distances.avg_l1_linf(a, b)
    sc.distances.hamming(a, b)
